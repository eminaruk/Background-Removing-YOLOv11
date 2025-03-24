import os
import random
import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

list_model = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]

# add choice for model selection 
print("Choose a model: ")
for i, model in enumerate(list_model):
    print(f"{i}: {model}")
model_choice = int(input("Enter the model number: "))
is_onnx = bool(int(input("Do you want to use ONNX? (0/1): ")))

model = YOLO(list_model[model_choice])

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]

if is_onnx and not os.path.exists(list_model[model_choice][:-3] + ".onnx"):
    model = model.export(format="onnx")
    model = YOLO(list_model[model_choice][:-3] + ".onnx")

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# define a video capture object 
vid = cv2.VideoCapture(0) 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if model_choice == 0 or model_choice == 1:
    person_class_id = 0
else:
    person_class_id = yolo_classes.index("person")  # Get the class ID for "person"

conf = 0.5

start_time = time.time()
frame_id = 0

while(True): 
    frame_id += 1
      
    # Capture the video frame 
    ret, frame = vid.read() 
    if not ret:
        break

    # Flip frame for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Check if a hand is detected
    hand_detected = result.multi_hand_landmarks is not None

    if not hand_detected:
        # Perform background removal if no hand is detected
        results = model.predict(frame, stream=True, conf=conf)
        
        # Filter masks for "person" class and remove background
        black_background = np.zeros(frame.shape, dtype=np.uint8)
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                if int(box.cls[0]) == person_class_id:  # If the class is "person"
                    points = np.int32([mask])
                    cv2.fillPoly(black_background, points, (255, 255, 255))
        
        frame = cv2.bitwise_and(frame, black_background)
    else:
        # Draw detected hands on the frame
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS : " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "Q to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
