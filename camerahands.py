import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd

# Load the trained model
model = load_model('smnist.h5')

# Initialize MediaPipe Hand Detection
initHand = mp.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mp.solutions.drawing_utils

# Function to extract finger tips
def fingers(landmarks):
    fingerTips = []
    tipIds = [4, 8, 12, 16, 20]
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0]-1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)

    for ids in range(1, 5):
        if landmarks[tipIds[ids]][2] < landmarks[tipIds[0] - 3][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)
    return fingerTips

# Function to preprocess and make predictions
def predict_sign(analysisframe):
    analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
    analysisframe = cv2.resize(analysisframe, (28, 28))

    nlist = []
    rows, cols = analysisframe.shape
    for i in range(rows):
        for j in range(cols):
            k = analysisframe[i, j]
            nlist.append(k)
    
    datan = pd.DataFrame(nlist).T
    colname = []
    for val in range(784):
        colname.append(val)
    datan.columns = colname

    pixeldata = datan.values
    pixeldata = pixeldata / 255
    pixeldata = pixeldata.reshape(-1, 28, 28, 1)
    prediction = model.predict(pixeldata)
    predarray = np.array(prediction[0])

    return predarray

# Open video capture device
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    h, w, c = frame.shape  # Capture frame dimensions

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        # 'q' pressed
        print("Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame horizontally

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mainHand.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            # Display hand landmarks
            draw.draw_landmarks(frame, handLMs, initHand.HAND_CONNECTIONS)
            
            # Determine which finger is up
            landmarkList = []
            for landmark in handLMs.landmark:
                landmarkList.append([landmark.x, landmark.y, landmark.z])
            finger = fingers(landmarkList)
            print("Fingers up:", finger)

            # Process sign language prediction
            x_center = 0
            y_center = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if x > x_center:
                    x_center = x
                if y < y_min:
                    y_min = y
                if y > y_center:
                    y_center = y
            y_min -= 20
            y_center += 20
            x_min -= 20
            x_center += 20 

            # Ensure the indices are within the bounds of the frame
            y_min = max(0, y_min)
            y_center = min(h, y_center)
            x_min = max(0, x_min)
            x_center = min(w, x_center)
            
            analysisframe = frame
            predarray = predict_sign(analysisframe[y_min:y_center, x_min:x_center])
            letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            predicted_sign = ""
            second_highest_confidence = 0
            second_predicted_sign = ""
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    print("Predicted Character 1: ", key)
                    print('Confidence 1: ', 100 * value)
                    predicted_sign = key
                elif value == high2:
                    print("Predicted Character 2: ", key)
                    print('Confidence 2: ', 100 * value)
                    if value > second_highest_confidence:
                        second_highest_confidence = value
                        second_predicted_sign = key
                elif value == high3:
                    print("Predicted Character 3: ", key)
                    print('Confidence 3: ', 100 * value)
            
            # Display predicted sign on the top-left corner
            cv2.putText(frame, "Predicted: " + second_predicted_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
