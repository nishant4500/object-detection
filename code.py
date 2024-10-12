import cv2
import pandas as pd
import numpy as np

# Threshold to detect object
thres = 0.45

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set the width
cap.set(4, 720)  # Set the height
cap.set(10, 70)  # Set the brightness

# Load class names from the coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Create a detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize an empty list to store detection logs
detection_log = []

while True:
    success, img = cap.read()
    if not success:
        break

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Log detections to pandas DataFrame
            detection_log.append({
                'class': classNames[classId - 1].upper(),
                'confidence': round(confidence * 100, 2),
                'box': box
            })

            # Draw rectangle and labels
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Save the detection logs to a pandas DataFrame
df = pd.DataFrame(detection_log)

# Display the logged detection data
print(df)

# Optionally, save the log to a CSV file
df.to_csv('detection_log.csv', index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()
