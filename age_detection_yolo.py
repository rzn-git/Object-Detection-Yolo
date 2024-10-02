from ultralytics import YOLO
import cv2

# Load YOLO model
yolo_model = YOLO("model/yolov5su.pt")  # Load YOLOv5 small model

# Load your age estimation model (hypothetical)
# age_model = load_age_model()  # This function would load your trained age model

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces with YOLO
    results = yolo_model(frame)
    
    for result in results:
        if result['name'] == 'person':  # Assuming 'person' is detected
            x1, y1, x2, y2 = result['bbox']  # Get bounding box
            
            # Crop face area for age estimation
            face = frame[y1:y2, x1:x2]
            # Estimate age
            # age = age_model.predict(face)  # Hypothetical prediction
            
            # Draw bounding box and age text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(frame, f'Age: {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
