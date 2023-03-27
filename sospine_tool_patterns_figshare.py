import os
import numpy as np
import pandas as pd
import itertools
from operator import itemgetter
import math
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import collections
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
from matplotlib.path import Path

verts = [
   (0., -4),  # left, bottom
   (0., 4),  # left, top
   (0.00001, 4),  # right, top
   (0.00001, -4),  # right, bottom
   (0., 0.),  # back to left, bottom
]

codes = [
    Path.MOVETO, #begin drawing
    Path.LINETO, #straight line
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY, #close shape. This is not required for this shape but is "good form"
]

path = Path(verts, codes)

# normalize the coordinates in the dataframe to [0,1] based on frame dimensions
def normalize_coords(data, frame_width, frame_height):
    data["x1"] = data["x1"] / frame_width
    data["x2"] = data["x2"] / frame_width
    data["y1"] = data["y1"] / frame_height
    data["y2"] = data["y2"] / frame_height

    return data


# reverse the normalization of the dataframe coordinates
def un_normalize_coords(data, frame_width, frame_height):
    data["x1"] = data["x1"] * frame_width
    data["x2"] = data["x2"] * frame_width
    data["y1"] = data["y1"] * frame_height
    data["y2"] = data["y2"] * frame_height

    return data


# Add empty columns to dataframe for frame info
def add_video_info(data, file_id, width, height, frames):
    data["trial_ID"] = []
    data["width"] = []
    data["height"] = []
    data["total_frames"] = []

    new_data = {"trial_ID": file_id, "width": width, "height": height, "total_frames": frames}
    data = data.append(new_data, ignore_index=True)

    return data

# calculates the number of times a tools went in and out of video frames
def calc_in_n_outs(data, search_thresh, tool):
    tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

    trials = list(tool_filtered["trial_id"].unique())

    ranges = []

    for trial in trials:

        trial_tool_filtered = tool_filtered.loc[tool_filtered["trial_id"] == trial]

        unique_instances = list(trial_tool_filtered["frame"].unique())

        for key, group in itertools.groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):
            group = list(map(itemgetter(1), group))
            group = list(map(int, group))
            ranges.append((group[0], group[-1]))

    return len(ranges)


# calculatess the area of the bounding box
def calc_bounding_box_area(x1, y1, x2, y2):
    h = y2 - y1 + 1
    w = x2 - x1 + 1

    return float(h * w)

# creates a list with the bounding box coordinates using a dataframe object
def get_bounding_box_list_df(tool_df):
    return [tool_df["x1"].iloc[0], tool_df["y1"].iloc[0], tool_df["x2"].iloc[0], tool_df["y2"].iloc[0]]


# creates a list with the bounding box coordinates using a row object
def get_bounding_box_list_row(tool_row):
    return [tool_row["x1"], tool_row["y1"], tool_row["x2"], tool_row["y2"]]

# returns the data with only high confidence detections and removes duplicate bboxes (IOU > 0.9)
def get_high_score_tools(data, tools, best_tool_thresholds):
    high_score_data = pd.DataFrame()

    for tool in tools:
        high_score_data = pd.concat(
            [high_score_data, data.loc[(data["label"] == tool) & (data["score"] >= float(best_tool_thresholds[tool]))]],
            ignore_index=True)

    high_score_data = high_score_data.sort_values(by=["trial_frame"])

    return high_score_data


# calculates the number of frames for a given trial using frame_to_trial_mapping.csv
def get_num_frames(trial_IDs):
    input_df = pd.read_csv("sospine_trial_outcomes.csv")

    frame_numbers = {}

    for ID in trial_IDs:
        try:
            frame_numbers[ID] = max(input_df.loc[input_df["trial_id"] == ID]["Total Frames at 1 FPS"].tolist())
        except:
            frame_numbers[ID] = 0

    return frame_numbers


# calculates the IOU value for 2 bounding boxes
def calc_iou(boxA, boxB):
    # if boxes dont intersect
    if do_boxes_intersect(boxA, boxB) is False:
        return 0
    interArea = get_Intersection_Area(boxA, boxB)
    union = get_Union_Area(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    return iou


# Checks if bounding boxes intersect
def do_boxes_intersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


# calculates the intersection area between 2 bounding boxes
def get_Intersection_Area(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if (do_boxes_intersect(b1, b2) == False):
        return 0.0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # A overlap
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


# calculates the union area for 2 bounding boxes
def get_Union_Area(boxA, boxB, interArea=None):
    area_A = calc_bounding_box_area(boxA[0], boxA[1], boxA[2], boxA[3])
    area_B = calc_bounding_box_area(boxB[0], boxB[1], boxB[2], boxB[3])
    if interArea is None:
        interArea = get_Intersection_Area(boxA, boxB)
    return float(area_A + area_B - interArea)


# calculates the best threshold for a tool using SOCAL ground truth detections
def find_best_thresholds(detections, truth, trial_IDs, tools_list, showGraphs=False):
    truth["trial_id"] = [x[0:-20] for x in truth["trial_frame"]]  # just the trial id
    truth["frame"] = [int(x[-13:-5]) for x in truth["trial_frame"]]  # just the frame number

    truth = truth[truth.trial_id.isin(trial_IDs)]
    truth.dropna(inplace=True)
    # truth.drop(["trial_frame"], axis = 1, inplace = True)

    print(tools_list)

    # result = calculate_metrics(detections, truth, trial_ID, tools_list, 0.5)

    results, best_tool_thresholds, tool_precisions = PlotPrecisionRecallCurve(detections, truth, trial_IDs, tools_list,
                                                                              IOUThreshold=0.5, showGraphic=showGraphs)

    return best_tool_thresholds, tool_precisions


# calculates average precision
def calc_avg_precision(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1 + i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + ((mrec[i] - mrec[i - 1]) * mpre[i])

    # ap = sum([mpre[i] for i in ii])/len(ii)  #????

    #return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

# Calculates metrics given the detections and ground truth data for a trial
def calculate_metrics(net_detections, truth, trial_ID, tools_list, IOUThreshold=0.50):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Get
    Args:
        boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
        (default value = 0.5);
        method (default = EveryPointInterpolation): It can be calculated as the implementation
        in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
        interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
        or EveryPointInterpolation"  (ElevenPointInterpolation);
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
        The keys of each dictionary are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Positive detections;
    """

    ret = []  # list containing metrics (precision, recall, average precision) of each class

    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates X,Y,X2,Y2)])
    groundTruths = []

    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = []

    # Get all classes
    classes = []

    for index, row in truth.iterrows():
        groundTruths.append([
            row["trial_frame"].replace(".jpeg", ".jpg"),
            row["label"], 1.0,
            get_bounding_box_list_row(row)
        ])

    for index, row in net_detections.iterrows():
        detections.append([
            row["trial_frame"],
            row["label"],
            row["score"],
            get_bounding_box_list_row(row),
        ])

    detections = sorted(detections, key=lambda conf: conf[2], reverse=True)

    for c in tools_list:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if (d[1] == c)]  # get only the detections for a specific tool

        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [
                    g]  # for each frame, creates gts dict with key=frame# and val=ground truths in that frame for the tool

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        thresholds = np.zeros(len(dects))

        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}

        #print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections

        vals = []
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))

            # Find ground truth image/frame number
            gt = gts[dects[d][0]] if dects[d][0] in gts else []

            iouMax = 0
            jmax = 0
            for j in range(len(gt)):  # for each ground truth annotation in a specific frame
                # print('Ground truth gt => %s' % (gt[j][3],))

                # print(dects[d], gt[j])
                iou = calc_iou(dects[d][3], gt[j][3])  # calculate IOU between each detection and each ground truth
                print(iou, dects[d][3], gt[j][3])

                # Find the detection bbox with the greatest overlap with the ground truth annotations being compared
                if (iou > iouMax):
                    iouMax = iou
                    jmax = j

            # print(dects[d][0], dects[d][1], iouMax, jmax)

            thresholds[d] = dects[d][2]

            # Assign detection as true positive/don't care/false positive
            if (iouMax > IOUThreshold):

                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
                    # print("TP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1

        # compute precision, recall and average precision

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        try:
            rec = np.divide(acc_TP, npos)  # tru pos / (tru pos + false neg)
            rec = np.append(rec, rec[len(rec) - 1])

            prec = np.divide(acc_TP, np.add(acc_FP, acc_TP))
            prec = np.append(prec, 0.0)
        except:

            rec = np.divide(acc_TP, npos)  # tru pos / (tru pos + false neg)
            prec = np.divide(acc_TP, np.add(acc_FP, acc_TP))

        # rec = np.append(rec, 1.0)

        false_neg = (npos - acc_TP)

        f1_score = 2 * np.divide(np.multiply(prec, rec), np.add(prec, rec))

        # Depending on the method, call the right implementation

        [ap, mpre, mrec, ii] = calc_avg_precision(rec, prec)
        # [ap, mpre, mrec, ii] = ElevenPointInterpolatedAP(rec, prec)

        # add class result in the dictionary to be returned. There are the calculates metrics for that tool
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'thresholds': thresholds,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'false positives': acc_FP,
            'true positives': acc_TP,
            'false negatives': false_neg,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'f1 score': f1_score
        }
        ret.append(r)

    return ret

def get_trial_test_set():
    return ['Clip0', 'S4A1', 'S8A2', 'S6A3'] #FOR sospine

# plots the Prec x Recall curve and returns the best confidence threshold for all tools
def PlotPrecisionRecallCurve(net_detections, truth, trial_IDs, tools_list, IOUThreshold=0.5, showAP=True,
                             showInterpolatedPrecision=False, savePath=None, showGraphic=True):
    # showGraphic = False
    # net_detections2 = net_detections.loc[net_detections["label"].isin(tools_list)]
    results = calculate_metrics(net_detections, truth, trial_IDs, tools_list, IOUThreshold)

    best_tool_thresholds = {}
    tool_precisions = {}

    # Each result represents a class
    for result in results:
        if result is None:
            raise IOError('Error: Class %d could not be found.')

        classId = result['class']
        precision = result['precision']  # average precision
        recall = result['recall']  # average recall
        thresholds = result['thresholds']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']

        npos = result['total positives']  # total real ground truth pos for that tool
        true_positives = result["true positives"]  # cumulative TPs for each threshold
        false_positives = result["false positives"]  # cumulative FPs for each threshold

        total_tp = result['total TP']
        total_fp = result['total FP']
        f1_score = result['f1 score']

        try:
            max_f1_score = np.nanmax(f1_score)
            max_f1_index = list(f1_score).index(max_f1_score)
            best_threshold = thresholds[max_f1_index]

            best_tool_thresholds[classId] = best_threshold

            # best_precision = true_positives[max_f1_index] / (true_positives[max_f1_index] + false_positives[max_f1_index])

            tool_precisions[classId] = average_precision

            # print(classId, "max f1 score:", max_f1_score, "best threshold: ", best_threshold, " best precision: ", best_precision)
            print(classId, "Average precision: ", average_precision)
            print(classId, total_tp, total_fp)
            print(classId, "best thresh: ", str(best_threshold)[0:6])

            if (showGraphic is True or savePath is not None):
                plt.close()

                #plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')

                bp_str = "{0:.2f}%".format(average_precision * 100)
                plt.plot(recall, precision, label="precision %s" % (str(bp_str)), linewidth=3.0)

                ax = plt.gca()
                ax.axhline(linewidth=2)
                ax.axvline(linewidth=2)

                plt.xlim(0, 1.0)
                plt.ylim(0, 1.0)

                #plt.plot(recall[max_f1_index], precision[max_f1_index], 'ro', label=('optimal thresh: ' + str(best_threshold)[0:6]))

                # plt.xlabel('Recall')
                # plt.ylabel('Precision')
                ax.set_xlabel('Recall', fontsize=14)
                ax.set_ylabel('Precision', fontsize=14)
                ax.tick_params(labelsize=12.0, length=5.0, width=2.0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2.0)  # change width

                if showAP:
                    # bp_str = "{0:.2f}%".format(best_precision * 100)

                    plt.title('Precision x Recall curve \nClass: %s  AP: %s' % (str(classId), (str(bp_str))), fontsize=16)
                else:
                    plt.title('Precision x Recall curve \nClass: %s' % (str(classId)), fontsize=16)

                #plt.legend(shadow=False)
                if savePath is not None:
                    plt.savefig(os.path.join(savePath, str(classId) + '.png'))
                if showGraphic is True:
                    plt.show()
                    # plt.waitforbuttonpress()
                    plt.pause(0.05)

        except Exception as e:
            print("no score for tool: ", classId)
            print(e)

            # **Default threshold for a tool if a best threshold cannot be determined (not enough instances or not present)
            best_tool_thresholds[classId] = 0.5
            tool_precisions[classId] = np.nan

    return results, best_tool_thresholds, tool_precisions

def calculate_tool_patterns(high_score_data, tools, trial_IDs, trial_frames_dict, bin_count=10):

    scaled_data = high_score_data.copy()

    scaled_frame = []

    for index, row in scaled_data.iterrows():
        scaled_frame.append(float(row["frame"]) / float(trial_frames_dict[row["trial_id"]]))

    scaled_data["scaled_frame"] = scaled_frame

    bins = list(np.linspace(0, 1, bin_count+1))

    trial_tool_totals_dict = {}
    trial_totals = {}

    trial_probs = {}
    tool_probs = {}
    for tool in tools:
        tool_probs[tool] = []
        trial_probs[tool] = {}

    for trial in trial_IDs:

        trial_tool_totals_dict[trial] = {}

        for index, tool in enumerate(tools):
            trial_tool_totals_dict[trial][tool] = [0 for i in range(0, len(bins) - 1)]
            trial_totals[tool] = 0

        new_scaled_data = scaled_data.loc[scaled_data["trial_id"] == trial]

        new_scaled_data["groups"] = pd.cut(new_scaled_data.scaled_frame, bins)

        for group_index, group in enumerate(list(new_scaled_data["groups"].unique())):

            filtered = new_scaled_data.loc[new_scaled_data["groups"] == group]

            tool_counts = filtered["label"].value_counts()

            for index, value in enumerate(tool_counts):

                if(tool_counts.index[index] in tools):

                    trial_tool_totals_dict[trial][tool_counts.index[index]][group_index] = value
                    trial_totals[tool_counts.index[index]] += value

        # for tool in trial_totals.keys():
        #     trial_probs[trial] = [np.divide(i, trial_totals[tool]) for i in trial_tool_totals_dict[trial][tool]]
        for tool in trial_totals.keys():
            trial_probs[tool][trial] = [ np.divide(i,trial_totals[tool]) for i in trial_tool_totals_dict[trial][tool]]

    values = {}

    for tool in tools:
        values[tool] = []
        for trial in trial_IDs:
            values[tool].append(trial_probs[tool][trial])

    tool_props_means = {}
    tool_props_std = {}

    for tool in tools:
        data = np.array(values[tool])
        tool_props_means[tool] = list(np.nanmean(data, axis=0))
        tool_props_std[tool] = list(np.nanstd(data, axis=0))

    bin_centers2 = [round(i + (1.0 / (2.0 * bin_count)), 2) for i in bins]
    bin_centers2 = bin_centers2[:-1]

    plt.title("tool proportions in bins averaged across trials")
    plt.xticks(np.arange(0.0, 1.0, (1.0 / bin_count)))

    for tool in tools:
        plt.plot(bin_centers2, tool_props_means[tool], label=tool)
        #plt.errorbar(bin_centers2, tool_props_means[tool], yerr=tool_props_std[tool])

    plt.legend(loc="upper left")
    plt.show()
    plt.clf()

        # for index,tool in enumerate(tools):
        #     plt.plot(bin_centers2, trial_probs[tool], label=tool)
        #     plt.title(trial)

        # plt.plot(bin_centers2, trial_probs["cottonoid"], label="cottonoid")
        # plt.plot(bin_centers2, trial_probs["muscle"], label="muscle")
        # plt.title(trial + " proportion of tool in each bin")
        # plt.xticks(np.arange(0.0, 1.0, (1.0/bin_count)))
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.clf()


def find_tool_patterns(data, tools, trial_IDs, trial_frames_dict, showGraphs=False):

    data = data.loc[data["label"].isin(tools)]
    data = data.sort_values(by=["trial_id"])

    outcomes = pd.read_csv("sospine_trial_outcomes.csv")
    outcomes = outcomes.loc[outcomes["trial_id"].isin(trial_IDs)].sort_values(by=["trial_id"])

    trial_IDs = list(outcomes["trial_id"].unique())

    ttr = list(outcomes["Time for repair"])
    success = list(outcomes["Success"])

    tth_scaled = []

    data = data.loc[data["trial_id"].isin(list(outcomes["trial_id"].unique()))].sort_values(by=["trial_id"])

    combs = []
    for i in range(1, len(tools) + 1):
        combs.append(list(itertools.combinations(tools, i)))

    combs.sort()

    combs_list = []
    for i in combs:
        for j in i:
            combs_list.append(list(j))

    combs_list.insert(0, ['empty'])

    new_dict = {}

    # 0 for no tools, 1+ for combinations of tools
    for i in range(0, len(combs_list)):
        new_dict[i] = []

    entropy_list = []
    cumu_divers = {}
    tool_combs_dict = {}

    for trial in trial_IDs:

        trial_data = data.loc[data["trial_id"] == trial]

        frame_entropy = []

        num_frames = trial_frames_dict[trial]

        for frame in range(1, num_frames + 1):  # frame in unique_frames:

            unique_labels = list(trial_data.loc[trial_data["frame"] == frame]["label"].unique())

            if (len(unique_labels) == 0):
                # print(trial, " empty ", frame)
                frame_entropy.append(0)

            val_combs = list(itertools.permutations(unique_labels, len(unique_labels)))

            # frame_lens.append(len(unique_labels))

            for index, i in enumerate(val_combs):
                if (list(i) in combs_list):
                    # print(combs_list.index(list(i))
                    frame_entropy.append(combs_list.index(list(i)))
                    break

        # trial_series = pd.Series(frame_entropy).value_counts().sort_index()
        # plt.bar(range(len(trial_series)), trial_series.values, align='center')
        # plt.xticks(range(len(trial_series)), trial_series.index.values, size='small')
        # plt.title(trial+" tool combs histogram")
        # plt.show()
        # plt.clf()

        tool_combs_dict[trial] = frame_entropy
        entropy_series = pd.Series(frame_entropy)
        counts = entropy_series.value_counts()  # this can be used to find patterns
        counts_index_list = list(counts.index.values)

        for i in range(0, len(combs_list)):

            if (i in counts_index_list):

                # normalize this to the length of trial due to conflicting correlations
                new_dict[i].append(counts.iloc[counts_index_list.index(i)] / trial_frames_dict[trial])
            else:
                new_dict[i].append(0)

        # print(trial)
        # print(counts.to_string())
        # print(list(entropy_series))
        probs = [i / len(entropy_series) for i in counts] # or entropy series or counts?

        entropy = stats.entropy(probs)
        entropy = entropy / math.log(len(entropy_series)) #normalize here by the log of the length of values
        entropy_list.append(entropy)

        cum_trial_diversity = []
        for i in range(1, len(entropy_series)):
            entropy_segment = entropy_series[0:i]
            segment_probs = [i / len(entropy_segment) for i in entropy_segment.value_counts()]
            segment_entropy = stats.entropy(segment_probs)
            segment_entropy = segment_entropy / math.log(len(entropy_segment))

            #uniq_combs = len(entropy_segment.unique()) # Cumulative cumulative
            #cum_trial_diversity.append(uniq_combs)
            if(np.isnan(segment_entropy)): segment_entropy = 0.0
            cum_trial_diversity.append(segment_entropy)

        cumu_divers[trial] = cum_trial_diversity

    if(showGraphs == True):
        for trial in trial_IDs:
            plt.plot(range(0, len(cumu_divers[trial])), cumu_divers[trial])
            plt.title(trial)
            plt.ylim(0.0,0.75)
            plt.show()
            plt.clf()

    combs_count_df = pd.DataFrame()

    for comb in new_dict.keys():
        combs_count_df[comb] = new_dict[comb]

    combs_count_df["entropy"] = entropy_list
    combs_count_df["trial"] = trial_IDs

    return combs_count_df, combs_list, trial_IDs, ttr, success, cumu_divers, tool_combs_dict

def gen_tool_combs_graphs(tool_combs_dict, combs_list, trial_IDs, showGraphs=True):

    dict = {"empty": "e", "muscle": "m", "cottonoid": "c", "suction": "sc", "grasper": "g", "string": "st" }

    new_combs_list = [[dict[y] for y in x] for x in combs_list]

    for trial in trial_IDs:

        trial_combs_df = pd.DataFrame()
        trial_combs_df["combs"] = tool_combs_dict[trial]
        trial_combs_df["frame"] = [x for x in range(1, len(tool_combs_dict[trial])+1)]

        if(showGraphs):
            plt.scatter(trial_combs_df["frame"], trial_combs_df["combs"], s=26, marker=path, c="black")
            plt.title(trial+" tool combs pattern across trial")
            plt.yticks(range(0, len(new_combs_list)), new_combs_list)

            ax = plt.gca()
            ax.set_xlabel('Trial Frames', fontsize=20)
            ax.set_ylabel('Possible Tool Combinations', fontsize=20)
            ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.0)  # change width

            plt.tight_layout()
            plt.show()
            plt.clf()

def get_ranges(unique_instances):

    groups = []

    for key, group in itertools.groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):
        group = list(map(itemgetter(1), group))
        group = list(map(int, group))

        groups.append(group)

    return groups


def smooth_labels(data, tool, threshold=10):

    data = data.loc[data["label"] == tool]

    trials = list(data["trial_id"].unique())

    dict = {}

    for trial in trials:

        trial_tool_filtered = data.loc[data["trial_id"] == trial]

        unique_instances = list(trial_tool_filtered["frame"].unique())

        groups = get_ranges(unique_instances)

        run = True
        index = 0

        if(len(groups) > 1):
            while(run):

                end = groups[index][-1]  #end of first
                beg = groups[index + 1][0]  #beginning of next

                if((beg-end) < threshold):

                    groups[index] = list( groups[index] + [i for i in range(end + 1, beg)] + groups[index + 1] )
                    groups.remove(groups[index + 1])
                    #index += 1

                else:
                    index += 1

                if(len(groups) == (index+1)):
                    run = False

        #print(tool, trial, groups)
        dict[trial] = groups

    return dict

def fill_tool_gaps(smooth_trials, data, tool):

    cut_data = data[["trial_frame", "frame", "label", "trial_id"]]

    new_tool_data = pd.DataFrame(columns=["trial_frame", "frame", "label", "trial_id"])
    removed_tool_data = pd.DataFrame(columns=["trial_frame", "frame", "label", "trial_id"])

    for trial in smooth_trials.keys():

        new_data = cut_data.loc[cut_data["trial_id"] == trial]

        missing_frames = []
        removal_frames = []

        for range in smooth_trials[trial]:

            for i in range:

                new_new_data = new_data.loc[new_data["frame"] == i]

                if (len(range) > 0): #***must be in the video for 3 sec at least

                    # if(tool == "nerve hook" and i == 619 and trial == "Clip0"):
                    #     print(new_new_data)

                    if(tool not in list(new_new_data["label"])):
                        missing_frames.append(i)

                else:
                    removal_frames.append(i)

        for frame in removal_frames:

            test = new_data.loc[ (new_data["frame"] == frame) & (new_data["label"] == tool) ]
            removed_tool_data = pd.concat([removed_tool_data, test], ignore_index=True).sort_index()

        extension = ".jpg"

        for frame in missing_frames:

            try:
                test = new_data.loc[new_data["frame"] == frame].iloc[0].copy()
                test["label"] = tool

                new_tool_data = pd.concat([new_tool_data,pd.DataFrame([test],columns=new_tool_data.columns)], ignore_index=True).sort_index()

                if(".jpeg" in test["trial_frame"]): extension = ".jpeg"

            except:

                num_zeros = 8 - sum(c.isdigit() for c in str(frame))
                num = "0"*num_zeros + str(frame)
                name = trial+"_frame_"+ num + extension
                test = [name, frame, tool, trial]

                new_tool_data = pd.concat([new_tool_data, pd.DataFrame([test],columns=new_tool_data.columns)], ignore_index=True).sort_index()

    return new_tool_data, removed_tool_data

def label_distributions(data, tools, trials, showFig=False, saveFig=False):

    group_dict = {}
    #tools.append("")
    for trial in trials:

        plt.figure().set_size_inches((9, 4))

        trial_data = data.loc[data["trial_id"] == trial]
        #trial_data.dropna(inplace=True)

        group_dict[trial] = {}

        for tool in tools:

            trial_tool_frames = list(trial_data.loc[trial_data["label"] == tool]["frame"])

            unique_instances = set(trial_tool_frames)
            unique_instances = list(unique_instances)
            unique_instances.sort()

            groups = get_ranges(unique_instances)
            group_dict[trial][tool] = groups

            #print(tool, len(groups)) #prints the number of ranges. correlations??

            if (len(trial_tool_frames) > 0):

                trial_tool_pres = [tool for i in range(0, len(trial_tool_frames))]
            else:
                trial_tool_pres = [tool]
                trial_tool_frames = [-10]

            plt.scatter(trial_tool_frames, trial_tool_pres, label=tool, s=200, marker=path)
            plt.title(trial + " - Tool Presence Distributions")
            # plt.xlim([-20, max(trial_tool_frames)+100])
            ax = plt.gca()
            ax.tick_params(axis='y', which='major', pad=1)
            ax.set_xlim(left=-10)
            ax.set_xlabel('Frames in trial', fontsize=20)
            ax.set_ylabel('Tools', fontsize=20)
            ax.tick_params(labelsize=14.0, length=5.0, width=2.0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.0)  # change width

            plt.tight_layout()

        if(saveFig):
            plt.rcParams['savefig.dpi'] = 400
            plt.savefig(trial + "_label_distribution.jpg")
            plt.show()
            plt.clf()

        if(showFig):
            plt.show()
            plt.clf()
    plt.clf()
    return group_dict

def dataset_overview(truth, trial_IDs_test, trial_IDs_training, tools):

    print("overview of dataset")

    truth_training = truth.loc[truth["trial_id"].isin(trial_IDs_training)]
    truth_test = truth.loc[truth["trial_id"].isin(trial_IDs_test)]

    print("training \n", truth_training["label"].value_counts())
    print("testing \n", truth_test["label"].value_counts())

    test_props = [val for val in truth_training["label"].value_counts().to_list()]
    train_props = [val for val in truth_test["label"].value_counts().to_list()]

    plt.bar(truth_training["label"].value_counts().index, truth_training["label"].value_counts().to_list())
    #plt.bar(truth_training["label"].value_counts().index, train_props)
    plt.title("training set overview")
    #plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.clf()

    plt.bar(truth_test["label"].value_counts().index, truth_test["label"].value_counts().to_list())
    #plt.bar(truth_test["label"].value_counts().index, test_props)
    plt.title("testing set overview")
    #plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.clf()

# generates all the APM for a give detections files for a single trial_id
def generate_APMs_from_detections_file(fileName, truthName, showGraphs=False):

    np.seterr(divide='ignore', invalid='ignore')

    data = pd.read_csv(fileName, names=["trial_frame", "frame", "x1", "y1", "x2", "y2", "score", "label", "trial_id"],
                       header=0)  # read in the input data file

    # -----------------------read ground truth to calculate confidence score threshold for tools
    truth = pd.read_csv(truthName, names=["trial_frame", "x1", "y1", "x2", "y2", "label"], header=0)
    truth.dropna(inplace=True)
    truth["trial_id"] = [i[0:-20] for i in truth["trial_frame"]]
    trial_IDs_training = [i for i in list(truth["trial_id"].unique()) if i not in get_trial_test_set()]  #training set only
    trial_IDs_test = [i for i in list(truth["trial_id"].unique()) if i in get_trial_test_set()]
    trial_IDs_truth = list(truth["trial_id"].unique())  # test set trial ids

    truth_test = truth.loc[truth["trial_id"].isin(trial_IDs_test)]  #filters to just test data.

    # total_frames = int(max(data["frame"]))

    all_tools = list(data["label"].unique())
    all_tools.sort()

    # list of tools to calculate metrics for
    tools = ['durotomy', 'grasper', 'needle driver', 'needle', 'nerve hook']  # drill
    instruments = ['grasper', 'needle driver', 'needle', 'nerve hook']
    tools.sort()

    trial_frames_dict = get_num_frames(list(truth["trial_id"].unique()))  # returns a dict of number of frames (val) for each trial (key)
    #dataset_overview(truth, trial_IDs_test, trial_IDs_training, tools)

    # get the best tool thresholds based on f1 score compared to ground truths
    print(trial_IDs_test, trial_IDs_training)
    best_tool_thresholds, tool_precisions = find_best_thresholds(data, truth, trial_IDs_test, tools, showGraphs=True)
    high_score_data = get_high_score_tools(data, tools, best_tool_thresholds)

    #print(best_tool_thresholds)

    #tool presence distributions
    # trial_groups_dict = label_distributions(truth, tools, trial_IDs_test, showFig=True, saveFig=False)
    trial_groups_dict = label_distributions(truth, tools, ["S8A2"], showFig=False, saveFig=True)
    trial_groups_dict = label_distributions(high_score_data, tools, ["S8A2"], showFig=False, saveFig=True)

def main():

    fileName = "yolov4_socal_sospine_detections_10.25_fixed.csv" #file of CV model detections (YOLOv4 in this case)
    truthName = "sospine.csv"

    generate_APMs_from_detections_file(fileName, truthName, showGraphs=False)


if __name__ == "__main__":
    main()