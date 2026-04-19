import cv2
import os

gesture_name = input("Enter Gesture Name: ")

path = f"dataset/{gesture_name}"

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)

    roi = frame[100:300,100:300]

    cv2.putText(frame, f"Images: {count}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{path}/{count}.jpg", roi)
        count += 1
        print("Saved:", count)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()