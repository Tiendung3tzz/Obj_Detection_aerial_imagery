import cv2
from sort import *
from ultralytics import YOLO
import cvzone
import math
import numpy as np

# Load the model
model = YOLO('yolo_weights/last3.pt')

# Define class names
classNames = ['Soldier', 'Tank', 'Truck_army']
color ={
                0:(255,0,0),
                1:(0,255,0),
                2:(0,0,255),
             }
# Initialize the video capture
cap = cv2.VideoCapture("video/fpv1.mp4")

# Initialize the video writer
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize the tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Limits for the line (not used in this example but kept for reference)
limits = [300, 400, 673, 297]

frame_id = 0

while True:
    success, img = cap.read()
    if not success:
        break

    height, width = img.shape[:2]
    new_width = int(width)
    new_height = int(height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    if frame_id % 1 == 0:
        results = model(img, stream=True)
        detections = np.empty((0, 5))
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                bbox = x1, y1, w, h
                
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                labelDec = ['Soldier', 'Tank', 'Truck_army']
                
                if (currentClass in labelDec) and (conf > 0.5):
                    cvzone.putTextRect(
                        img, 
                        f'{classNames[cls]} - {conf}', 
                        (max(0, x1), max(35, y1 - 20)), 
                        scale=1,
                        thickness=2,
                        offset=7,                        
                    )
                    cvzone.cornerRect(img, bbox, l=8, rt=5,colorR=color[cls])
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
        
        resultsTracker = tracker.update(detections)
    print(f"Processed frame {frame_id}/{total_frames}")

    out.write(img)  # Write the frame to the video file
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
