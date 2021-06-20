import cv2
import numpy as np

# It is used to load the Deep Learning network to the system when the OpenCV library is used in Python
net = cv2.dnn.readNet('E:/Helmet and License Plate Detection/yolov3/yolov3_training_last.weights', 'E:/Helmet and License Plate Detection/yolov3/yolov3_testing.cfg')

classes = []
with open("E:/Helmet and License Plate Detection/yolov3/classes.txt", "r") as f:
   classes = f.read().splitlines()

cap = cv2.VideoCapture('E:/Helmet and License Plate Detection/yolov3/test_full_vid.mp4')
cap = cv2.VideoCapture('E:/Helmet and License Plate Detection/yolov3/test_full_vid.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
counter = 0

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            
            
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            

            if(label == "license_plate"):
                license_plate = img[y:(y + h), x:(x + w)]
                # cv2.imshow('Image', license_plate)
                # cv2.waitKey(0)

            
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()