from ultralytics import YOLO
import cv2
import math
from sort import *
import numpy as np
from scipy.ndimage import gaussian_filter

carvideo = cv2.VideoCapture("football.mp4")
model = YOLO("Yolo-Weights/best2.pt")

classNames = ["person", "player", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
              "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask2.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Time-related variables
start_time = cv2.getTickCount()

# List to store player positions
player_positions = []

# Define the codec and create a VideoWriter object
output_path = "heat_deneme3.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(carvideo.get(3)), int(carvideo.get(4))))

while True:
    success, img = carvideo.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    resultsTracker = tracker.update(detections)

    # Calculate elapsed time since the start of the video
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    
    # Extract player positions within the pitch
    current_player_positions = []

    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        current_player_positions.append((cx, cy))
        
        #text = f'ID:{int(track_id)}'
        #text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=1)[0]
        #rect_size = (text_size[0] + 10, text_size[1] + 10)
        #rect_start = (x1 - 2, y1 - text_size[1] - 5)
        #rect_end = (x1 - 2 + rect_size[0], y1)

        #cv2.rectangle(img, rect_start, rect_end, (0, 255, 0), thickness=cv2.FILLED)
        #cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=1, color=(255, 0, 0))
        #cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


    # Append current player positions to the list
    player_positions.append(current_player_positions)

    # Generate heat map
    heat_map = np.zeros((50, 100), dtype=np.float32)
    for positions in player_positions:
        for x, y in positions:
            # Normalize positions to fit the pitch dimensions
            normalized_x = int((x / img.shape[1]) * 100)
            normalized_y = int((y / img.shape[0]) * 50)
            heat_map[normalized_y, normalized_x] += 1

    # Apply Gaussian filter to smooth the heat map
    heat_map_smoothed = gaussian_filter(heat_map, sigma=2)

    # Normalize the heat map values
    heat_map_normalized = (heat_map_smoothed / heat_map_smoothed.max() * 255).astype(np.uint8)

    # Apply colormap to the heat map
    heat_map_colored = cv2.applyColorMap(heat_map_normalized, cv2.COLORMAP_JET)

    # Resize the heat map to match the dimensions of the original frame
    heat_map_resized = cv2.resize(heat_map_colored, (img.shape[1], img.shape[0]))

    # Blend the heat map with the original frame
    result = cv2.addWeighted(img, 0.7, heat_map_resized, 0.3, 0)

    cv2.imshow("Video", result)
    # Write frame to the output video
    output_video.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and close all windows
output_video.release()
carvideo.release()
cv2.destroyAllWindows()