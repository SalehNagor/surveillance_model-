import numpy as np 
import cv2 
from ultralytics import YOLO  
import supervision as sv  
import time  

# Load YOLO model
model = YOLO("yolov8s.pt")  

# Initialize tracking and annotation tools
tracker = sv.ByteTrack()  
box_annotator = sv.BoxAnnotator()  
label_annotator = sv.LabelAnnotator()  
trace_annotator = sv.TraceAnnotator() 
heat_map_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,  # Heatmap position
    opacity=0.5,  
    radius=25,  
    kernel_size=25,  
    top_hue=0,  
    low_hue=125,  
)

# Variables to track statistics
frame_count = 0
total_processing_time = 0

def callback(frame: np.ndarray, display_mode: int) -> np.ndarray:
    global total_processing_time, frame_count

    start_time = time.time()  # Record the start time for processing

    results = model(frame)[0]  
    # Detect the objects.
    detections = sv.Detections.from_ultralytics(results) 
    detections = tracker.update_with_detections(detections)  

    unique, counts = np.unique(detections.class_id, return_counts=True)  
    class_count = dict(zip(unique, counts))  # Create a dictionary of class_id:count
    object_count = len(detections.tracker_id)  # Count total number of objects

    if display_mode == 0:  # Normal tracking mode
        labels = [
            f"#{tracker_id} {results.names[class_id]}"  # Create label with tracker ID and object name
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

    elif display_mode == 1:  # Heatmap mode
        annotated_frame = heat_map_annotator.annotate(scene=frame.copy(), detections=detections)

    elif display_mode == 2:  # Head counting mode
        annotated_frame = frame.copy()
        for i in range(len(detections.tracker_id)):
            if detections.class_id[i] == 0:  # Check if object is a person (class_id = 0)
                x1, y1, x2, y2 = detections.xyxy[i]  # Get bounding box coordinates
                head_y = int(y1)  # Top y-coordinate of the bounding box
                center_x = int((x1 + x2) / 2)  # Center x-coordinate
                cv2.circle(annotated_frame, (center_x, head_y), 5, (0, 0, 255), -1)  # Draw red dot on head

    # Calculate processing time and update statistics
    processing_time = time.time() - start_time  # Time taken to process the frame
    total_processing_time += processing_time
    frame_count += 1

    # Display total object count
    cv2.putText(
        annotated_frame,
        f"Total Count: {object_count}",  # Display total number of objects
        (10, 30),  # Text position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        0.8,  # Font size
        (0, 255, 0),  # Text color (green)
        2  # Font thickness
    )

    # Display the count for each object class
    offset = 90  # Vertical offset for text
    for class_id, count in class_count.items():  # Loop through class counts
        label = f"{results.names[class_id]}: {count}"  
        cv2.putText(
            annotated_frame,
            label,
            (10, offset),  
            cv2.FONT_HERSHEY_SIMPLEX,  
            0.8, 
            (0, 255, 0),  
            2  
        )
        offset += 20  

    return annotated_frame  


# Load the video
cap = cv2.VideoCapture("/workspaces/Computer_Vision/Vedioes/Hajj_vedio.mp4")  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Save the results in this path.
output_path = "/workspaces/Computer_Vision/Results/Result_Vedio.mp4"  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

display_mode = 0  
last_switch_time = 0  
switch_interval = 3

# Call callback() to proccessing the veadio frame by frame.
while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break  

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  
    if current_time - last_switch_time >= switch_interval:
        display_mode = (display_mode + 1) % 3  
        last_switch_time = current_time  

    annotated_frame = callback(frame, display_mode)

    out.write(annotated_frame)  

cap.release() 
out.release()  
print("Video saved successfully at:", output_path)

# Calculate and display average processing time statistics
average_processing_time = total_processing_time / frame_count if frame_count > 0 else 0
print(f"Total Frames: {frame_count}")
print(f"Total Processing Time: {total_processing_time:.2f} seconds")
print(f"Average Processing Time per Frame: {average_processing_time:.4f} seconds")