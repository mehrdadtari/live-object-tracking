import cv2
import numpy as np
from yolo_detector import yolo_detector
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(detections, tracks: List[STrack]):
    detections = np.array(detections)
    if not np.any(detections[:,:4]) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections[:,:4])
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# # Create tracker object
tracker = BYTETracker(BYTETrackerArgs())

# Create detector model
model = YOLO("yolov8m.pt")

# Retrieving Class Names and IDs
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# class_ids of interest - car, motorcycle, bus and truck
# CLASS_ID = [2, 3, 5, 7]
# class_ids of interest - person
CLASS_ID = [0]

# Mimimum object detection accuracy to consider an object. Between 0 and 1.
DETECT_ACC = 0.5

# Select video stream or webcam.
# cap = cv2.VideoCapture("car.mp4")
cap = cv2.VideoCapture(0)

# If you want to use a region of interest (roi), use with_roi = True
with_roi = False

while True:
    ret, frame = cap.read()
    if frame is not None:
        height, width, _ = frame.shape
    
    if not ret:
        break

    # Extract Region of interest
    # set the percentage of height and width for roi.
    if with_roi:
        roi = frame[int(0.4 * height): int(0.7 * height), int(0 * width): int(1 * width)]
    else:
        roi = frame

    # 1. Object Detection
    roi, detections = yolo_detector(roi, model, CLASS_ID, DETECT_ACC)

    # 2. Object Tracking
    if detections:
        tracks = tracker.update(output_results=np.array(detections),
                img_info=roi.shape,
                img_size=roi.shape)

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)

        for track_id, box_id in zip(tracker_id, detections):
            x, y, x2, y2 = box_id[:4]
            if track_id:
                cv2.putText(roi, str(track_id), (x, y - 26), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 2)
    
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()