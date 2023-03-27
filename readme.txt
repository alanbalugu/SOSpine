##############
## OVERVIEW ##
##############


Simulated Outcomes for Durotomy Repair in Minimally Invasive Spine Surgery (SOSpine) dataset version 1.

See official dataset publication here:
https://figshare.com/projects/Simulated_Outcomes_for_Durotomy_Repair_in_Minimally_Invasive_Spine_Surgery_SOSpine_/142508


#################
## DATASET USE ##
#################


This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. 
https://creativecommons.org/licenses/by-nc/4.0/


If you plan to publish work that used this dataset in any capacity, please cite this data release.


These are neurosurgical training exercise videos performed on a perfusion-based cadaveric models. These simulations contain no patient data, no protected health information, and are not derived from patient care settings. All videos are owned by the Department of Neurosurgery at the University of Southern California. Please email any correspondence to dandonohomd@gmail.com


#########################
## DATASET DESCRIPTION ##
#########################


Repairing an incidental durotomy to prevent continued cerebrospinal fluid (CSF) leak is an important surgical skill for a spine surgeon. Larger durotomies generally require primary closure with sutures to repair the defect, and improper techniques can lead to patient morbidity. Given possible consequences, more experienced surgeons may be performing this procedure, depriving surgical trainees of adequate experience. Incidental durotomies may occur in 1% to 15.9% of surgical cases, depending on the case, and proper repair must ensure reliable collocation of the two dural edges with tight closure and minimal CSF leakage [1]. A previously validated perfusion-based cadaveric simulator was used to teach neurosurgery interns how to close a large durotomy using minimally-invasive surgery (MISS) techinques [2, 3]. For each training exercise, a laminectomy was performed on a fresh cadaver to expose the spinal dura in the cervical region, and a 12-gauge arterial catheter was inserted into the subdural-subarachnoid space to reconstitute CSF pressures using the Minneti method of perfusion. Subsequently, the entire dural was exposed along the thoracolumbar spine and a MEDTRx (Medtronic Sofamor Danek, Medtronic Inc, Dublin, Ireland) tubular retractor was positioned in place. A 1 cm durotomy was then created to simulate a defect, and the perfusion was changed to a pulsation flow of CSF to simulate an intraoperative scenario. Participants were then required to use a 22 mm x 6 cm tube for dural repair under surgical loupe magnification using finetip tissue forceps, Castroviejo needle holders, and 6-0 Prolene suture (Ethicon, Somerville, New Jersey) [3]. Success in this procedure involved tying two sutures to close the durotomy tightly without a subsequence CSF leak after a simulated Valsalva maneuver with higher CSF pressures. Further details of the methods and procedures can be found in the associated publication [3]. 


Neurosurgery residents of all years were recruited, and the participants were given a pre-test to determine prior MISS experience and confidence levels in performing a CSF leak repair. The participants were also allowed to practice their technique before the study trials before performing the durotomy repair task in three successive trials [3]. Non-identifiable demographic information about the participants were collected in addition to total time to complete the durotomy and quality of surgical repair.


Intraoperative video was taken during each of the participant trials. The dataset that is presented here includes the individual frames from the videos of these trials after appropriate segmentation, down-sampling, and annotation for selected surgical instruments. The data was processed based upon a previously described approach and prepared accordingly for dissemination to the research community [4]. Videos were recorded at a frame rate of 30 frames per second (fps) and at a resolution of 1920 x 1080 pixels. Video was downsamplied from 30 fps to 1 fps using FFmpeg for the dataset presented here [3, 5].


Each video frame was hand-annotated for surgical tools using tool-tip annotations. After the video was downsampled to 1 fps, the frames were annotated using the open-sourced image annotation software VoTT [6]. The tip and base of instruments were marked for: needle, needle driver. Key points were marked for nerve hook and grasper.[a][b] The two ends of the durotomy were similarly marke[c]d. In addition to video recordings of the surgical durotomy repair trials, task outcomes (e.g. time for durotomy repair, task success), and demographic data (post-graduate year, prior experience with MISS, number of prior cases) was recorded for each participant.


A complete description of the dataset files and columns can be found in the DATASET FILES section below.


[1] Espiritu MT, Rhyne A, Darden BV 2nd. Dural tears in spine surgery. J Am Acad
Orthop Surg. 2010;18(9):537-545.
[2] Zada G, Bakhsheshian J, Pham M, et al. Development of a Perfusion-Based Cadaveric Simulation Model Integrated into Neurosurgical Training: Feasibility Based On Reconstitution of Vascular and Cerebrospinal Fluid Systems. Oper Neurosurg (Hagerstown). 2018;14(1):72-80. doi:10.1093/ons/opx074
[3] Buchanan IA, Min E, Pham MH, et al. Simulation of dural repair in minimally invasive spine surgery with the use of a perfusion-based cadaveric model. Oper Neurosurg. 2019;17(6):616-621. doi:10.1093/ons/opz041 
[4] D. J. Pangal, G. Kugener, S. Shahrestani, F. Attenello, G. Zada, and D. A. Donoho, "Technical Note: A Guide to Annotation of Neurosurgical Intraoperative Video for Machine Learning Analysis and Computer Vision," World Neurosurg., Published online March 12, 2021, doi:10.1016/j.wneu.2021.03.022.
[5] “FFmpeg.” [Online]. Available: https://ffmpeg.org/. [Accessed: 03-Dec-2020].
[6] microsoft/VoTT. Microsoft, 2020.




########################
## DATASET STATISTICS ##
########################


Number of surgeons: 8
Number of trials with outcomes data: 24
Number video annotated trials: 24
Number of annotated frames: 15,698
Total tool-tip annotations: 53,238 
Total bounding box annotation: 21,371


Number of tool-tips annotated


Needle Tip: 1,571
Needle Base: 347
Needle Driver Tip: 4,544
Needle Driver Base: 9,371
Grasper: 8,478
Nerve Hook: 4,648
Durotomy: 24,279


Number of generated bounding boxes:


Needle: 1,417
Needle Driver: 4,238
Grasper: 2,375
Nerve Hook: 1,200
Durotomy: 12,141


###################
## DATASET FILES ##
###################


##############
# frames.zip #
##############


Description:
Zip file that contains all of the video frames, as .jpeg files.

File located on FigShare: https://figshare.com/articles/dataset/frames_zip/20201636


#############
# sospine_tool_tips.csv #
#############


Description: 
        Each row in this file corresponds to a single tool-tip annotation in a single frame. If a frame has multiple annotated tools, then there will be a rows for each tool in view. If a frame has no tools in view, then there will be a single row for this frame where the first column contains the frame file name and the remaining columns are empty. 


        For the coordinate values, the top left corner of the image is considered the origin (0,0). The coordinates are integer pixel values.


trial_frame
        Frame file name beginning with the trial name and ending with frame number


x1
        The left x coordinate


y1
        The top y coordinate


x2
        The right x coordinate


y2
        The bottom y coordinate


label
        The tool tip label. Can be one of 7 labels: needle tip, needle base, durotomy, nerve hook, grasper, needle driver tip, or needle driver [d]base


#############
# sospine_bbox.csv #
#############


Description: 
        This file contains bounding boxed generated programmatically using the tool-tip annotations previously described in this dataset release. Each row in this file corresponds to a single tool bounding box annotation in a single frame. If a frame has multiple annotated tools, then there will be a rows for each tool in view. If a frame has no tools in view, then there will be a single row for this frame where the first column contains the frame filename and the remaining columns are empty. Bounding boxes were generated by finding the minimum and maximum of the x and y coordinates for all points annotated for a given tool to create a rectangular bounding box. Tools with only one tool-tip annotation were simply given a square-shaped bounding box by expanding the singular annotation by 50 pixels in either direction. 


        For the coordinate values, the top left corner of the image is considered the origin (0,0). The coordinates are integer pixel values.


trial_frame
        Frame file name beginning with the trial name and ending with frame number


x1
        The left x coordinate


y1
        The top y coordinate


x2
        The right x coordinate


y2
        The bottom y coordinate


label
        The bounding box label. Can be one of 5 labels: needle, durotomy, nerve hook, grasper, or needle driver[e]




##############################
# sospine_outcomes.csv #[f][g][h]
##############################


Description:
        This file contains the participant level demographics data for participating surgeons, results of the repair trials, and information about the associated videos. Missing values indicate either unavailable participant data or unavailable video footage of those trials.


Trial ID        
        Trial ID that refers to the surgeon and attempt number. S# denoting surgeon and A# denoting the attempt. 


Length        
        Length of the video in minutes and seconds. 


Postgraduate year
        Participant’s year along neurosurgery residency


Prior experience with MISS
        Whether participant had experience with minimally invasive spine surgery cases in the past        


Number prior cases
        Number of minimally invasive spine surgery (MISS) cases that the participant has been involved with prior to the SoSpine trials
        
Time for repair
        Number of seconds needed to complete the durotomy repair procedure. Defined from the time from initial instrument contact with dura to the time at which instruments exited the field on securing the final knot


Leak At 40mmHg
        Presence of a CSF leak after completion of the durotomy repair and pressurizing the system to 40 mm Hg.


Total Frames at 1 FPS
        Number of video frames from the trial that are included in this dataset within the frames.zip file.
