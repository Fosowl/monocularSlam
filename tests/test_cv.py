import cv2

video_path = '../videos/forest_road.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break
    cv2.imshow('Video', frame)
    key = cv2.waitKey(25)
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()
