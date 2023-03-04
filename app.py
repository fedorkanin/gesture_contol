import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils


# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
# capture from webcam 1. If it doesn't work, try 0
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which MediaPipe uses)
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the resulting image
        cv2.imshow('MediaPipe Hand Recognition', frame)

    # Exit program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
