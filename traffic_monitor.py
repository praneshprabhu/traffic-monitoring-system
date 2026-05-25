import cv2
from ultralytics import YOLO

print("=" * 50)
print("TRAFFIC MONITORING SYSTEM")
print("=" * 50)

# Load model
print("\n[1/3] Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("✓ Model loaded!")

# Open camera
print("[2/3] Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    print("Trying camera 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("ERROR: No camera found!")
    exit()

print("✓ Camera opened!")

# Vehicle classes
vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
total_count = 0

# Tracking for counting
tracked = {}  # Stores vehicle IDs to avoid double counting
next_id = 0

print("[3/3] Starting monitor...")
print("\n" + "=" * 50)
print("HOW TO USE:")
print("  • Vehicles are counted when they cross the GREEN line")
print("  • Press 'q' - Quit")
print("  • Press 'r' - Reset counters")
print("=" * 50 + "\n")

# Get frame height for counting line
ret, frame = cap.read()
if ret:
    frame_height = frame.shape[0]
    line_y = int(frame_height * 0.7)  # Line at 70% down the screen
else:
    line_y = 400  # Default

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        print("Warning: Failed to get frame")
        break
    
    # Flip for mirror effect (optional)
    frame = cv2.flip(frame, 1)
    
    # Update line position if frame height changed
    current_height = frame.shape[0]
    if current_height != frame_height:
        line_y = int(current_height * 0.7)
        frame_height = current_height
    
    # Draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 3)
    cv2.putText(frame, "COUNTING LINE", (frame.shape[1]//2 - 80, line_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Run detection
    results = model(frame, verbose=False)
    
    # Process each detection
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                
                # Only process vehicles
                if cls in vehicle_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    vehicle_type = vehicle_classes[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw center point
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Add label
                    cv2.putText(frame, vehicle_type, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Create a unique ID for this vehicle
                    vehicle_id = f"{vehicle_type}_{center_x}_{center_y}_{y1}"
                    
                    # Check if vehicle is crossing the line
                    if vehicle_id not in tracked:
                        # New vehicle - check if it's crossing the line
                        if center_y > line_y - 15 and center_y < line_y + 15:
                            # Vehicle is crossing the line!
                            tracked[vehicle_id] = True
                            counts[vehicle_type] += 1
                            total_count += 1
                            
                            # Show counting feedback
                            cv2.putText(frame, f"+1 {vehicle_type}!", 
                                       (x1, y1 - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            print(f"✓ {vehicle_type} counted! Total: {total_count}")
                    else:
                        # Vehicle already counted, check if it's still near line
                        if abs(center_y - line_y) > 50:
                            # Vehicle moved away, remove from tracking
                            del tracked[vehicle_id]
    
    # Remove old tracked vehicles (cleanup)
    to_remove = []
    for vid in tracked:
        found = False
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_y = (y1 + y2) // 2
                        test_id = f"{vehicle_classes[cls]}_{(x1+x2)//2}_{center_y}_{y1}"
                        if test_id == vid:
                            found = True
                            break
                if found:
                    break
        if not found:
            to_remove.append(vid)
    
    for vid in to_remove:
        del tracked[vid]
    
    # Display statistics on screen
    y = 30
    cv2.putText(frame, f"TOTAL: {total_count}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    for vehicle, count in counts.items():
        y += 35
        cv2.putText(frame, f"{vehicle}: {count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit | 'r' to reset", 
               (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "Vehicles count when crossing GREEN line", 
               (10, line_y - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Traffic Monitor - Counting System', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Quitting...")
        break
    elif key == ord('r'):
        total_count = 0
        counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        tracked = {}
        print("\n[INFO] Counters reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "=" * 50)
print("FINAL TRAFFIC REPORT")
print("=" * 50)
print(f"Total vehicles counted: {total_count}")
print("\nBreakdown by type:")
for vehicle, count in counts.items():
    if count > 0:
        print(f"  {vehicle}: {count}")
print("=" * 50)

