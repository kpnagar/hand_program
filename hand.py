import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


logo = cv2.imread('ring.png')
size = 100
logo = cv2.resize(logo, (50, 20))

# Get Image dimensions
img_height, img_width, _ = logo.shape


# # Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

# For webcam input:
cap = cv2.VideoCapture(0)


# Get frame dimensions
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )

# Print dimensions
print('image dimensions (HxW):',img_height,"x",img_width)
print('frame dimensions (HxW):',int(frame_height),"x",int(frame_width))

x = 50
y = 50

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # add image to frame
    # image[ y:y+img_height , x:x+img_width ] = logo

    roi = image[ y:y+img_height , x:x+img_width ]
    roi[np.where(mask)] = 0
    roi += logo

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    # print(dir(results),"---results------")
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      # print((results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]),"--------- WRIST ------")
      print((results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]),"--------- INDEX_FINGER_TIP ------")
      print((results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]),"--------- MIDDLE_FINGER_TIP ------")
      print(results.multi_handedness[0].classification[0].label,"--------- Hand ------")
      # print((results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]),"--------- High ------")
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()