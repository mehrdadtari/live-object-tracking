# Object Tracking with YOLOv8 and ByteTrack
Using this repo, you can detect and track 80 class types in a video file or live stream. 
This repo has 2 parts:
1. Object Detection with **YOLOv8**
2. Object Tracking with **ByteTrack**

In the main.py, you can manually select the class IDs that you want to track in a video. For your convenience, all the existing class IDs and their corresponding class names are printed in the terminal upon execution. 

Also, you can manually set the minimum detection accuracy to consider for tracking.

In addition, you have the capability to do object tracking in the entire frame or select a region of interest (roi) to perform object tracking. 

In each frame, for the detected objects, the bounding box, the class name, the detection accuracy, and the tracking ID are illustrated. 
