import onnxruntime as rt
import cv2
import numpy as np

# Load the ONNX model
sess = rt.InferenceSession('runs/train/exp3/weights/best.onnx')

# Define the input shape of the model
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_dtype = sess.get_inputs()[0].type

# Define the output shape of the model
output_name = sess.get_outputs()[0].name
output_shape = sess.get_outputs()[0].shape
output_dtype = sess.get_outputs()[0].type

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
    outputs = sess.run([output_name], {input_name: img.astype(np.float32)})[0]

    # Parse the model output and draw the bounding boxes and labels
    for output in outputs:
        for detection in output:
            conf = detection[4]
            if conf > 0.5:
                x, y, w, h = detection[:4] * 640
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                label = detection[5]
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









