#!/usr/bin python3

import cv2 as cv
from sources.slam import Slam

video = cv.VideoCapture('./videos/forest_road.mp4')
if not video.isOpened():
    print("Failed to read video")
    exit()

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_dim = (width, height)

slam = Slam(width, height)

while True:
    ret, frame = video.read()
    assert ret
    frame = cv.resize(frame, video_dim, interpolation = cv.INTER_AREA)
    slam.view_points(frame)
    cv.imshow('Video', frame)
    key = cv.waitKey(25)
    if key == ord('q'):
        break
video.release()
cv.destroyAllWindows()
