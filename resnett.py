import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv
import math
import time

# COCO dataset class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Load a pretrained Faster R-CNN model with ResNet50 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

# Initialize DeepSort
tracker = DeepSort(max_age=75, n_init=3, nms_max_overlap=1.0)

# Video paths
input_video_path = '/notebooks/20240724T191501.mkv'  # Change to your video path
output_video_path = '20240724T191501/20240724T191501_output_video.mp4'

# Load the video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize variables for tracking
previous_positions = {}
total_distance_travelled = {}
ball_angles = {}
csv_data = []

# Assume 1 pixel = 1/154 meters
meters_per_pixel = 1 / 154

# Process the video in batches of 25 frames
batch_size = 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Dictionary to store frame timestamps
frame_timestamps = {}

with torch.no_grad():
    for batch_start in tqdm(range(0, total_frames, batch_size), desc="Processing Video"):
        batch_end = min(batch_start + batch_size, total_frames)

        # Initialize variables for the batch
        batch_images = []
        batch_detections = []

        # Read and process frames in the batch
        for frame_idx in range(batch_start, batch_end):
            ret, frame = cap.read()
            if not ret:
                break

            # Capture timestamp for each frame
            frame_timestamps[frame_idx] = time.time()

            # Convert frame to tensor
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
            batch_images.append(image_tensor)

        # Check if batch_images is not empty before concatenating
        if len(batch_images) > 0:
            # Perform object detection on the batch
            batch_images = torch.cat(batch_images, dim=0)
            batch_predictions = model(batch_images)

            # Process the batch predictions
            for i in range(len(batch_predictions)):
                frame_idx = batch_start + i
                predictions = batch_predictions[i]
                boxes = predictions['boxes'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()

                # Filter labels for "person", "sports ball", and "tennis racket"
                filtered_indices = []
                for j, label in enumerate(labels):
                    if label < len(COCO_INSTANCE_CATEGORY_NAMES):
                        if COCO_INSTANCE_CATEGORY_NAMES[label] in ['person', 'sports ball', 'tennis racket']:
                            filtered_indices.append(j)
                    else:
                        print(f"Warning: Label {label} is out of range for COCO_INSTANCE_CATEGORY_NAMES.")

                filtered_boxes = boxes[filtered_indices]
                filtered_scores = scores[filtered_indices]
                filtered_labels = labels[filtered_indices]

                # Prepare detections for DeepSort
                detections = []
                for j in range(len(filtered_boxes)):
                    if filtered_scores[j] > 0.5:
                        x1, y1, x2, y2 = filtered_boxes[j]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append((bbox, filtered_scores[j], filtered_labels[j]))

                batch_detections.append(detections)

        # Update tracker with current detections
        for i, detections in enumerate(batch_detections):
            frame_idx = batch_start + i
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure frame is read at the correct timestamp
            while time.time() < frame_timestamps[frame_idx]:
                time.sleep(0.001)

            tracked_objects = tracker.update_tracks(detections, frame=frame)

            # Draw bounding boxes and labels on the frame
            for obj in tracked_objects:
                if not obj.is_confirmed():
                    continue

                x1, y1, x2, y2 = map(int, obj.to_tlbr())
                label = COCO_INSTANCE_CATEGORY_NAMES[obj.det_class]
                track_id = obj.track_id

                # Convert pixel coordinates to meters
                x_meters = ((x1 + x2) / 2) * meters_per_pixel
                y_meters = ((y1 + y2) / 2) * meters_per_pixel
                z_meters = 1.0 if label == 'sports ball' else 0.0

                # Calculate the ball speed if it's a sports ball
                if label == 'sports ball':
                    current_position = (x_meters, y_meters)
                    if track_id in previous_positions:
                        prev_position = previous_positions[track_id]
                        distance_meters = math.sqrt((current_position[0] - prev_position[0]) ** 2 +
                                                    (current_position[1] - prev_position[1]) ** 2)
                        speed_m_per_s = distance_meters * fps
                        speed_km_per_h = speed_m_per_s * 3.6

                        # Calculate angle of movement
                        dx = current_position[0] - prev_position[0]
                        dy = current_position[1] - prev_position[1]
                        angle = math.degrees(math.atan2(dy, dx))
                        if angle < 0:
                            angle += 360

                        ball_angles[track_id] = angle

                        # Determine trajectory
                        trajectory = "down the line"
                        if 0 <= angle < 20 or 120 <= angle < 180:
                            trajectory = "cross court"

                    else:
                        speed_km_per_h = 0
                        trajectory = "down the line"

                    previous_positions[track_id] = current_position

                # Calculate distance traveled by players
                if label == 'person':
                    current_position = (x_meters, y_meters)
                    if track_id in previous_positions:
                        prev_position = previous_positions[track_id]
                        distance_meters = math.sqrt((current_position[0] - prev_position[0]) ** 2 +
                                                    (current_position[1] - prev_position[1]) ** 2)
                        total_distance_travelled[track_id] = total_distance_travelled.get(track_id, 0) + distance_meters

                    previous_positions[track_id] = current_position

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prepare text and background for ID
                text = f'{label} ID: {track_id}'
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw the background rectangle for the text
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)  # Filled rectangle

                # Draw the text on top of the background
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if label == 'sports ball':
                    speed_text = f'Speed: {speed_km_per_h:.2f} km/h'
                    (speed_text_width, speed_text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    
                    # Draw background for speed text
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + speed_text_width, y2 + speed_text_height + 15), (255, 0, 0), -1)

                    # Draw the speed text
                    cv2.putText(frame, speed_text, (x1, y2 + speed_text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    trajectory_text = f'Trajectory: {trajectory}'
                    (trajectory_text_width, trajectory_text_height), _ = cv2.getTextSize(trajectory_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Draw background for trajectory text
                    cv2.rectangle(frame, (x1, y2 + speed_text_height + 20), (x1 + trajectory_text_width, y2 + speed_text_height + trajectory_text_height + 30), (255, 0, 0), -1)

                    # Draw the trajectory text
                    cv2.putText(frame, trajectory_text, (x1, y2 + speed_text_height + trajectory_text_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store the information in CSV data
                csv_data.append([frame_idx, track_id, label, x_meters, y_meters, z_meters,
                                 speed_km_per_h if label == 'sports ball' else '',
                                 total_distance_travelled.get(track_id, '') if label == 'person' else '',
                                 trajectory if label == 'sports ball' else ''])

            # Write the processed frame to the output video
            out.write(frame)

# Write CSV data to file
with open('20240724T191501/20240724T191501_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'ID', 'Label', 'X (m)', 'Y (m)', 'Z (m)', 'Speed (km/h)', 'Distance Traveled (m)', 'Trajectory'])
    writer.writerows(csv_data)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()