import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
from tqdm import tqdm   
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv
import math
import os
import torch.multiprocessing as mp  # Use torch.multiprocessing
from time import time
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
def process_video_segment(segment_path, output_path, frame_idx, gpu_id):
    # Set the device for the current process
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model onto the assigned GPU
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Initialize tracker and transform
    tracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=1.0)
    transform = T.Compose([T.ToTensor()])

    # Initialize video capture and writer
    cap = cv2.VideoCapture(segment_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    target_width = 640
    target_height = 360
    previous_frame = None

    # Initialize variables for tracking
    previous_positions = {}
    total_distance_travelled = {}
    ball_angles = {}
    csv_data = []

    # Process the video in batches of 25 frames
    batch_size = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time()
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

                # Resize the frame to target resolution
                resized_frame = cv2.resize(frame, (target_width, target_height))

                # Convert frame to tensor
                image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

                # Calculate frame difference if previous frame exists
                if previous_frame is not None:
                    frame_diff = cv2.absdiff(resized_frame, previous_frame)
                    frame_diff_percentage = np.mean(frame_diff) / 255 * 100
                    if frame_diff_percentage < 95:
                        # If the difference is less than 95%, skip detection
                        continue

                # Update previous frame
                previous_frame = resized_frame.copy()

                # Add frame to batch for detection
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

                    # Resize the frame to target resolution
                    resized_frame = cv2.resize(frame, (target_width, target_height))

                    tracked_objects = tracker.update_tracks(detections, frame=resized_frame)

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
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Prepare text and background for ID
                        text = f'{label} ID: {track_id}'
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                        # Draw the background rectangle for the text
                        cv2.rectangle(resized_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)  # Filled rectangle

                        # Draw the text on top of the background
                        cv2.putText(resized_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        if label == 'sports ball':
                            speed_text = f'Speed: {speed_km_per_h:.2f} km/h'
                            (speed_text_width, speed_text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                            # Draw background for speed text
                            cv2.rectangle(resized_frame, (x1, y2 + 5), (x1 + speed_text_width, y2 + speed_text_height + 15), (255, 0, 0), -1)

                            # Draw the speed text
                            cv2.putText(resized_frame, speed_text, (x1, y2 + speed_text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            trajectory_text = f'Trajectory: {trajectory}'
                            (trajectory_text_width, trajectory_text_height), _ = cv2.getTextSize(trajectory_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                            # Draw background for trajectory text
                            cv2.rectangle(resized_frame, (x1, y2 + speed_text_height + 20), (x1 + trajectory_text_width, y2 + speed_text_height + trajectory_text_height + 30), (255, 0, 0), -1)

                            # Draw the trajectory text
                            cv2.putText(resized_frame, trajectory_text, (x1, y2 + speed_text_height + trajectory_text_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Store the information in CSV data
                        csv_data.append([frame_idx, track_id, label, x_meters, y_meters, z_meters,
                                         speed_km_per_h if label == 'sports ball' else '',
                                         total_distance_travelled.get(track_id, '') if label == 'person' else '',
                                         trajectory if label == 'sports ball' else ''])

                    # # Write the processed frame to the output video
                    # out.write(resized_frame)                    # Store the information in CSV data
                    #     csv_data.append([frame_idx, track_id, label, x_meters, y_meters, z_meters])

            # Write the processed frame to the output video
            out.write(resized_frame)
            frame_idx += 1  # Increment frame index

    end_time = time()
    processing_time = end_time - start_time
    print(f"Segment processing time: {processing_time:.2f} seconds on GPU {gpu_id}")

    # Release resources
    cap.release()
    out.release()

    return processing_time, csv_data

def split_video(input_path, segment_length=450):  # Adjusted segment length for better balance
    segment_paths = []
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frame_count = segment_length * fps

    for i in range(0, total_frames, segment_frame_count):
        segment_path = f"segment_{i//segment_frame_count}.mp4"
        segment_paths.append(segment_path)
        out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        for j in range(segment_frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()
    return segment_paths

def merge_videos(segment_paths, output_path):
    clips = []
    for segment in segment_paths:
        print(f"Processing segment: {segment}")
        try:
            clip = VideoFileClip(segment)
            clips.append(clip)
        except KeyError as e:
            print(f"Warning: Unable to retrieve FPS for {segment}. Skipping this segment. Error: {e}")
        except Exception as e:
            print(f"Error loading {segment}: {e}")

    if clips:
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec='libx264')
    else:
        print("No valid clips to merge.")

def main(input_video_path, output_video_path):
    # Step 1: Split the video
    segment_paths = split_video(input_video_path)

    # Step 2: Process each segment concurrently on different GPUs
    output_segment_paths = [f"output_{segment}" for segment in segment_paths]
    processing_times = []
    csv_data = []

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(process_video_segment, [(segment_paths[i], output_segment_paths[i], 0, i % num_gpus) for i in range(len(segment_paths))])
        processing_times = [t for t, _ in results]
        csv_data = [data for _, data in results]

    total_processing_time = sum(processing_times)
    print(f"Total processing time: {total_processing_time:.2f} seconds")

    all_csv_data = []
    for segment_csv_data in csv_data:
        all_csv_data.extend(segment_csv_data)

    # Ensure the output directory exists before saving CSV
    output_dir = os.path.dirname(output_video_path)
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, 'tracking_data.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Track ID', 'Label', 'X (m)', 'Y (m)', 'Z (m)', 'Speed (km/h)', 'Distance Traveled (m)', 'Trajectory'])
        writer.writerows(all_csv_data)

    # Step 3: Merge the output segments
    merge_videos(output_segment_paths, output_video_path)

    # Clean up segment files if needed
    for segment in segment_paths + output_segment_paths:
        os.remove(segment)

if __name__ == '__main__':
    t_start_time = time()
    input_video_path = '/notebooks/ByteTrack/20240725T201500.mkv'
    output_video_path = '20240725T201500/mul3_output.mp4'
    main(input_video_path, output_video_path)
    t_end_time = time()
    t_processing_time = t_end_time - t_start_time
    print(f"entire procesing time {t_processing_time}")
