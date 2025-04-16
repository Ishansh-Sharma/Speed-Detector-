import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
import os

# Constants
VIDEO_PATH = '/Users/ishanshsharma/Vihaan2.0/2165-155327596_small.mp4'
OVERSPEED_THRESHOLD_KMPH = 40
REAL_WORLD_DISTANCE_METERS = 100
FRAME_INTERVAL = 5

model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

line1_y = int(frame_height * 0.4)
line2_y = int(frame_height * 0.6)

vehicle_tracker = {}
overspeeders = []

# Output folders
os.makedirs("output/crops", exist_ok=True)
output_csv = open('output/overspeeders.csv', 'w', newline='')
csv_writer = csv.writer(output_csv)
csv_writer.writerow(['Vehicle_ID', 'Speed_kmph', 'Frame', 'Vehicle_Type'])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_type = model.names[cls_id]
        vehicle_id = f"{cls_id}_{x1}_{y1}"
        cy = int((y1 + y2) / 2)

        if vehicle_id in vehicle_tracker:
            prev_frame, prev_cy = vehicle_tracker[vehicle_id]
            if frame_count - prev_frame >= FRAME_INTERVAL:
                pixel_distance = abs(cy - prev_cy)
                time_elapsed = (frame_count - prev_frame) / fps
                pixel_per_meter = abs(line2_y - line1_y) / REAL_WORLD_DISTANCE_METERS
                real_distance = pixel_distance / pixel_per_meter
                speed_mps = real_distance / time_elapsed
                speed_kmph = speed_mps * 3.6

                if speed_kmph > OVERSPEED_THRESHOLD_KMPH:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{round(speed_kmph, 1)} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Save cropped image
                    crop = frame[y1:y2, x1:x2]
                    crop_filename = f"output/crops/{vehicle_id}_{frame_count}.jpg"
                    cv2.imwrite(crop_filename, crop)

                    # Log to CSV
                    overspeeders.append({
                        "id": vehicle_id,
                        "speed": round(speed_kmph, 2),
                        "frame": frame_count,
                        "type": vehicle_type
                    })
                    csv_writer.writerow([vehicle_id, round(speed_kmph, 2), frame_count, vehicle_type])

        vehicle_tracker[vehicle_id] = [frame_count, cy]

    # Draw reference lines
    cv2.line(frame, (0, line1_y), (frame_width, line1_y), (255, 0, 0), 2)
    cv2.line(frame, (0, line2_y), (frame_width, line2_y), (255, 0, 0), 2)

    cv2.imshow("Overspeed Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
output_csv.close()
cv2.destroyAllWindows()
