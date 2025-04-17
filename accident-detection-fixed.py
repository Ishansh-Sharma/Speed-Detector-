import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort # Import the SORT tracker
import datetime
import time  # Fixed import (was 'timepip')

# --- Configuration ---
VIDEO_SOURCE = "traffic.mp4"  # Path to your video file OR 0 for webcam
MODEL_PATH = 'yolov8n.pt'    # YOLOv8 nano model - fast, good for CPU. Use yolov8s.pt etc. for more accuracy if GPU available.
CONFIDENCE_THRESHOLD = 0.4   # Minimum confidence score for detections
RELEVANT_CLASSES = [2, 3, 5, 7] # COCO Class IDs for: car, motorcycle, bus, truck
                                # You might need to adjust these based on the specific YOLO model/dataset

# --- Accident Detection Parameters ---
MIN_STATIONARY_TIME_SECONDS = 5.0 # How long an object must be stationary to trigger an alert
STATIONARY_VELOCITY_THRESHOLD = 1.5 # Pixels per frame threshold to consider an object stationary
ALERT_COOLDOWN_SECONDS = 10.0 # Minimum time between alerts for the SAME object

# --- Initialization ---
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

print("Initializing tracker...")
mot_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) # Adjust SORT parameters as needed
print("Tracker initialized.")

print(f"Opening video source: {VIDEO_SOURCE}...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    exit()
print("Video source opened.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# Dictionary to store tracking information (center points, timestamps, stationary status)
tracked_objects = {}
# Dictionary to store last alert time for each object ID to manage cooldown
alert_cooldown = {}

frame_count = 0

# --- Main Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_count += 1
    current_time_s = time.time() # Use system time for duration calculations

    # --- 1. Object Detection ---
    results = model(frame, stream=True, verbose=False) # stream=True is efficient

    detections_for_sort = []
    detected_classes = {} # Store class for each detection bbox

    for result in results:
        boxes = result.boxes # Boxes object for bbox outputs
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
            conf = float(box.conf[0])            # Confidence score
            cls = int(box.cls[0])                # Class ID

            # Filter by confidence and relevant classes
            if conf > CONFIDENCE_THRESHOLD and cls in RELEVANT_CLASSES:
                detections_for_sort.append([x1, y1, x2, y2, conf])
                # Store class temporarily using the bounding box itself as a key (will match later)
                detected_classes[tuple([x1, y1, x2, y2])] = model.names[cls] # Get class name

    detections_np = np.array(detections_for_sort) if detections_for_sort else np.empty((0, 5))

    # --- 2. Object Tracking ---
    # Update SORT tracker
    # Input: np.array([[x1, y1, x2, y2, score], ...])
    # Output: np.array([[x1, y1, x2, y2, track_id], ...])
    tracked_bbs_ids = mot_tracker.update(detections_np)

    # --- 3. Accident Logic Data Collection & Analysis ---
    current_tracked_ids = set()
    for track in tracked_bbs_ids:
        x1_t, y1_t, x2_t, y2_t, track_id = map(int, track)
        current_tracked_ids.add(track_id)
        center_x = (x1_t + x2_t) // 2
        center_y = (y1_t + y2_t) // 2

        # Find the original detection class for this track
        # (This is a simple matching based on bbox, could be improved with IoU if needed)
        obj_class = "unknown"
        for det_box, cls_name in detected_classes.items():
            if det_box[0] == x1_t and det_box[1] == y1_t and det_box[2] == x2_t and det_box[3] == y2_t:
                 obj_class = cls_name
                 break # Found the match

        if track_id not in tracked_objects:
            # New object tracked
            tracked_objects[track_id] = {
                'history': [(center_x, center_y, current_time_s)],
                'stationary_start_time': None,
                'is_stationary': False,
                'class': obj_class
            }
        else:
            # Existing object - Fixed the history data structure issue
            history = tracked_objects[track_id]['history']
            if history:  # Check if history exists and is not empty
                last_x, last_y, last_time = history[-1]  # Get the last point from history
                tracked_objects[track_id]['history'].append((center_x, center_y, current_time_s))
                tracked_objects[track_id]['class'] = obj_class # Update class just in case

                # Keep history buffer limited (optional)
                if len(tracked_objects[track_id]['history']) > 50:
                     tracked_objects[track_id]['history'].pop(0)

                # Calculate velocity (simple Euclidean distance over time)
                time_diff = current_time_s - last_time
                if time_diff > 0: # Avoid division by zero
                    distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                    velocity = distance / time_diff # Pixels per second

                    # Check for stationary state (using simple pixel distance per frame is often easier)
                    if distance < STATIONARY_VELOCITY_THRESHOLD: # Check distance moved since *last frame*
                        if not tracked_objects[track_id]['is_stationary']:
                            # Just became stationary
                            tracked_objects[track_id]['is_stationary'] = True
                            tracked_objects[track_id]['stationary_start_time'] = current_time_s
                    else:
                        # Is moving
                        tracked_objects[track_id]['is_stationary'] = False
                        tracked_objects[track_id]['stationary_start_time'] = None
                else:
                    # No time difference, assume stationary for this check
                    if not tracked_objects[track_id]['is_stationary']:
                        tracked_objects[track_id]['is_stationary'] = True
                        tracked_objects[track_id]['stationary_start_time'] = current_time_s
            else:
                # If history is empty for some reason, just add current point
                tracked_objects[track_id]['history'] = [(center_x, center_y, current_time_s)]

        # --- 4. Alerting Logic ---
        if tracked_objects[track_id]['is_stationary'] and tracked_objects[track_id]['stationary_start_time'] is not None:
            stationary_duration = current_time_s - tracked_objects[track_id]['stationary_start_time']
            # Check if stationary duration exceeds threshold AND cooldown period has passed
            if stationary_duration >= MIN_STATIONARY_TIME_SECONDS:
                last_alert_time = alert_cooldown.get(track_id, 0)
                if current_time_s - last_alert_time > ALERT_COOLDOWN_SECONDS:
                    # --- !!! ALERT TRIGGERED !!! ---
                    alert_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("-" * 30)
                    print(f"🚨 ALERT! Possible Incident Detected 🚨")
                    print(f"  Time: {alert_time_str}")
                    print(f"  Object ID: {track_id}")
                    print(f"  Class: {tracked_objects[track_id]['class']}")
                    print(f"  Location (Center): ({center_x}, {center_y})")
                    print(f"  Stationary Duration: {stationary_duration:.2f} seconds")
                    print("-" * 30)

                    # Update cooldown timestamp
                    alert_cooldown[track_id] = current_time_s

                    # --- TODO: Implement actual alerting mechanism here ---
                    # E.g., send webhook, API call, save snapshot
                    # snapshot_filename = f"alert_snapshot_{track_id}_{alert_time_str.replace(':','-').replace(' ','_')}.jpg"
                    # cv2.imwrite(snapshot_filename, frame)
                    # print(f"  Snapshot saved: {snapshot_filename}")
                    # --------------------------------------------------------

                    # Mark as alerted to potentially change visualization
                    # (Add an 'alerted' flag to tracked_objects if needed)


    # --- Cleanup old tracks ---
    lost_ids = set(tracked_objects.keys()) - current_tracked_ids
    for lost_id in lost_ids:
        # print(f"Track {lost_id} lost.")
        del tracked_objects[lost_id]
        if lost_id in alert_cooldown:
            del alert_cooldown[lost_id]


    # --- 5. Visualization (Optional) ---
    vis_frame = frame.copy()
    for track in tracked_bbs_ids:
        x1_t, y1_t, x2_t, y2_t, track_id = map(int, track)
        label = f"ID:{track_id} {tracked_objects[track_id]['class']}"
        color = (0, 255, 0) # Green for moving

        if tracked_objects[track_id]['is_stationary']:
             color = (0, 165, 255) # Orange for stationary
             if tracked_objects[track_id]['stationary_start_time'] is not None:  # Added safety check
                 stationary_duration = current_time_s - tracked_objects[track_id]['stationary_start_time']
                 label += f" Stop:{stationary_duration:.1f}s"
                 if stationary_duration >= MIN_STATIONARY_TIME_SECONDS and track_id in alert_cooldown and current_time_s - alert_cooldown[track_id] < ALERT_COOLDOWN_SECONDS:
                      color = (0, 0, 255) # Red if recently alerted

        cv2.rectangle(vis_frame, (x1_t, y1_t), (x2_t, y2_t), color, 2)
        cv2.putText(vis_frame, label, (x1_t, y1_t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS
    proc_time = time.time() - current_time_s
    current_fps = 1.0 / (proc_time + 1e-9) # Add small epsilon to avoid division by zero
    cv2.putText(vis_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Accident Detection Pipeline", vis_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Resources released.")
