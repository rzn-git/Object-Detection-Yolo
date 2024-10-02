from ultralytics import YOLO
import cv2

# Load YOLOv5 model (this will automatically download pretrained weights)
model = YOLO("yolov5s.pt")  # You can use 'yolov5m.pt', 'yolov5l.pt' for larger models

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 on the frame
    results = model(frame)

    # Convert results to OpenCV format (with bounding boxes)
    annotated_frame = results[0].plot()

    # Display the output frame
    cv2.imshow("YOLOv5 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
