from draw_bbox import bbox_drawer

def yolo_detector(frame, model, CLASS_ID, DETECT_ACC):
    # yolov8m is the medium version. 
    # We also have smaller and bigger models. Bigger models are more accurate but slower.

    # results is a list with one entry of type:
    # ultralytics.yolo.engine.results.Results object
    results = model(frame, device="mps")
    # result is ultralytics.yolo.engine.results.Results object from
    # results list.
    result = results[0]

    processed_frame, detections = bbox_drawer(result, frame, model, CLASS_ID, DETECT_ACC)

    return processed_frame, detections

