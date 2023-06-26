import cv2
import torch
import numpy as np

# Load the model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/weights/best.pt')

# Define the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the image and resize it
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1) / 255.0
    img = img.reshape(1, 3, 640, 640)

    # Run the model
    results = model(img)

    # Parse the model output and draw the bounding boxes and labels
    for result in results.xyxy[0]:
        conf = result[4]
        if conf > 0.5:
            x1, y1, x2, y2 = result[:4]
            label = result[5]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLOv5 Real-time Detection", frame)

    # Press q to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()



