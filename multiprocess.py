import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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

    # Initialize variables for tracking
    previous_positions = {}
    total_distance_travelled = {}
    csv_data = []
    meters_per_pixel = 1 / 154  # Assume 1 pixel = 1/154 meters

    # Process each frame
    start_time = time()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Perform object detection
            predictions = model(image_tensor)

            # Process the predictions
            boxes = predictions[0]['boxes'].cpu()
            if predictions and 'labels' in predictions[0]:
                labels = predictions[0]['labels'].cpu()
                scores = predictions[0]['scores'].cpu()

                # Filter labels for "person", "sports ball", and "tennis racket"
                filtered_indices = [i for i, label in enumerate(labels) if COCO_INSTANCE_CATEGORY_NAMES[label] in ['person', 'sports ball', 'tennis racket']]
                filtered_boxes = boxes[filtered_indices].numpy()
                filtered_scores = scores[filtered_indices].numpy()
                filtered_labels = labels[filtered_indices].numpy()

                # Prepare detections for DeepSort
                detections = []
                for i in range(len(filtered_boxes)):
                    if filtered_scores[i] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = filtered_boxes[i]
                        bbox = [x1, y1, x2-x1, y2-y1]  # DeepSort expects (x, y, w, h)
                        detections.append((bbox, filtered_scores[i], filtered_labels[i]))

                # Update tracker with current detections
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
                    z_meters = 1.0 if label == 'sports ball' else 0.0  # Constant z for simplicity

                    # Store the information in CSV data
                    csv_data.append([frame_idx, track_id, label, x_meters, y_meters, z_meters])

            # Write the processed frame to the output video
            out.write(frame)
            frame_idx += 1  # Increment frame index

    end_time = time()
    processing_time = end_time - start_time
    print(f"Segment processing time: {processing_time:.2f} seconds on GPU {gpu_id}")

    # Release resources
    cap.release()
    out.release()

    return processing_time, csv_data

def split_video(input_path, segment_length=600):  # Adjusted segment length for better balance
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
    clips = [VideoFileClip(segment) for segment in segment_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec='libx264')

def main(input_video_path, output_video_path):
    # Step 1: Split the video
    segment_paths = split_video(input_video_path)

    # Step 2: Process each segment concurrently on different GPUs
    output_segment_paths = [f"output_{segment}" for segment in segment_paths]
    processing_times = []
    csv_data = []

    # Get the available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(process_video_segment, [(segment_paths[i], output_segment_paths[i], 0, i % num_gpus) for i in range(len(segment_paths))])
        processing_times = [t for t, _ in results]
        csv_data = [data for _, data in results]

    total_processing_time = sum(processing_times)
    print(f"Total processing time: {total_processing_time:.2f} seconds")

    # Concatenate CSV data
    all_csv_data = []
    for segment_csv_data in csv_data:
        all_csv_data.extend(segment_csv_data)

    # Save CSV data
    csv_file = 'mul3_tracking_data.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Track ID', 'Label', 'X (m)', 'Y (m)', 'Z (m)'])
        writer.writerows(all_csv_data)

    # Step 3: Merge the output segments
    merge_videos(output_segment_paths, output_video_path)

    # Clean up segment files if needed
    for segment in segment_paths + output_segment_paths:
        os.remove(segment)

if __name__ == '__main__':
    input_video_path = '/notebooks/20240627T110047.mkv'
    output_video_path = 'mul3_output.mp4'
    main(input_video_path, output_video_path)
