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
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
]

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Load a pretrained Faster R-CNN model with ResNet50 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

# Initialize DeepSort
tracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=1.0)

# Video paths
input_video_path = '/notebooks/data/20240725T203000.mkv'
#/notebooks/ball_detection/Code_Padel/20240725T203000_scaled.mp4'
output_video_path = '20240725T203000_idfix_boutput_video.mp4'

# Load the video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"width is {width} height is {height}")
# Initialize variables for tracking
previous_positions = {}
total_distance_travelled = {}
csv_data = []
meters_per_pixel = 1 / 135

# Define the regions for ID assignment
regions = {
    1: np.array([[916, 83], [622, 83], [1304, 133], [1297, 492]]),
    2: np.array([[1304, 83], [1297, 83], [1704, 144], [1977, 498]]),
    3: np.array([[622, 494], [72, 1437], [1297, 492], [1274, 1438]]),
    4: np.array([[1297, 492], [1274, 1438], [1977, 498], [2481, 1435]]),
}

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Dictionary to store frame timestamps
frame_timestamps = {}
# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Assign ID based on location
# Update code to include position-based ID assignment
def assign_id_based_on_position(current_position, previous_positions, assigned_ids):
    """ Assign an ID based on position in the specified segments and closeness to the last known position """
    x, y = current_position
    segment_id = None

    # Define the segments and corresponding IDs
    segments = {
        1: [(916, 83), (1304, 83), (1297, 492), (622, 494)],
        2: [(1304, 83), (1704, 83), (1977, 498), (1297, 492)],
        3: [(622, 494), (72, 1437), (1274, 1438), (1297, 492)],
        4: [(1297, 492), (1274, 1438), (2481, 1435), (1977, 498)]
    }

    # Check in which segment the person is
    for seg_id, corners in segments.items():
        if is_point_in_polygon((x, y), corners):
            segment_id = seg_id
            break

    # If there's a segment found
    if segment_id is not None:
        if segment_id not in assigned_ids:
            # Assign the ID based on the segment if not assigned already
            return segment_id
        else:
            # If the segment ID is already assigned, calculate closeness to the previous position
            return assign_closest_id(current_position, previous_positions, assigned_ids, segment_id)
    return None

def is_point_in_polygon(point, polygon):
    """ Check if a point is inside a polygon using the ray-casting algorithm """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def assign_closest_id(current_position, previous_positions, assigned_ids, segment_id):
    """ Assign the closest available ID based on proximity to previous positions """
    x, y = current_position
    available_ids = set(range(1, 5)) - assigned_ids

    # Find the closest previous position among available IDs
    min_distance = float('inf')
    closest_id = None
    for available_id in available_ids:
        if available_id in previous_positions:
            prev_x, prev_y = previous_positions[available_id]
            distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_id = available_id

    # Assign the closest available ID
    return closest_id if closest_id is not None else segment_id

# Modified tracking loop

frame_skip_rate = 12  # For example, to skip 12 frames

with torch.no_grad():
    for frame_idx in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every nth frame
        if frame_idx % frame_skip_rate != 0:
            continue

        frame_timestamps[frame_idx] = time.time()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        prediction = model(image_tensor)
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        filtered_indices = [j for j, label in enumerate(labels) 
                            if label < len(COCO_INSTANCE_CATEGORY_NAMES) and 
                            COCO_INSTANCE_CATEGORY_NAMES[label] == 'person' and
                            scores[j] > 0.5]

        detections = []
        for j in filtered_indices:
            x1, y1, x2, y2 = boxes[j]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, scores[j], labels[j]))

        tracked_objects = tracker.update_tracks(detections, frame=frame)
        assigned_ids = set()

        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, obj.to_tlbr())
            current_position = ((x1 + x2) / 2, (y1 + y2) / 2)
            x_pixel=current_position[0]
            y_pixel=current_position[1]
            pixel_to_meter=meters_per_pixel
            # Convert pixel coordinates to meters
            #print (f"x is {current_position[0]} and y is {current_position[1]}")
            if 83 <= y_pixel < 213:
                x_meters = x_pixel * pixel_to_meter*1.59
                y_meters = y_pixel * pixel_to_meter* 5.87
            elif 213 <= y_pixel < 498:
                x_meters = x_pixel * pixel_to_meter*1.195
                y_meters = y_pixel * pixel_to_meter*3.32
            elif 498 <= y_pixel < 842:
                x_meters = x_pixel * pixel_to_meter*0.85
                y_meters = y_pixel * pixel_to_meter*1.37
            elif 842 <= y_pixel < 1259:
                x_meters = x_pixel * pixel_to_meter*0.66
                y_meters = y_pixel * pixel_to_meter* 1.13
            elif 1259 <= y_pixel < 1435:
                x_meters = x_pixel * pixel_to_meter*0.59
                y_meters = y_pixel * pixel_to_meter*1.13
            else:
                x_meters = current_position[0] * meters_per_pixel
                y_meters = current_position[1] * meters_per_pixel

            # Assign an ID based on location and previous positions
            track_id = assign_id_based_on_position(current_position, previous_positions, assigned_ids)

            if track_id is not None:
                assigned_ids.add(track_id)

                # Calculate distance traveled for this ID
                if track_id in previous_positions:
                    prev_position = previous_positions[track_id]
                    distance_meters = math.sqrt((x_meters - prev_position[0]) ** 2 +
                                                (y_meters - prev_position[1]) ** 2)
                    total_distance_travelled[track_id] = total_distance_travelled.get(track_id, 0) + distance_meters

                previous_positions[track_id] = (x_meters, y_meters)

                label = COCO_INSTANCE_CATEGORY_NAMES[obj.det_class]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the ID above the bounding box
                text = f'{label} ID: {track_id}'
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1) 
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Append the required details to the CSV data
                
                csv_data.append([frame_idx, track_id, label,x_meters,y_meters,'', total_distance_travelled.get(track_id, ''),x_pixel , y_pixel])
                print(f"{frame_idx} of id: {track_id},  {label},  at: {x_meters},{y_meters} travelled:  {total_distance_travelled.get(track_id, '')} and x_pixel is {x_pixel} y is {y_pixel} ")
        # Write the processed frame to the output video 
        out.write(frame)
        

# Write CSV data to file 
with open('20240725T203000_scaledxy_person_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'ID', 'Label', 'X (m)', 'Y (m)', 'Z (m)', 'Distance Traveled (m)','x_pixel','y_pixel'])
    writer.writerows(csv_data)

# Release resources 
cap.release()
out.release()
cv2.destroyAllWindows()
