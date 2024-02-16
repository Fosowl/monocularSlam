#!/usr/bin python3

import cv2 as cv
from sources.slam import Frame
from sources.slam import Slam
from sources.render import Renderer3D
import time

video = cv.VideoCapture('./videos/forest_road.mp4')
if not video.isOpened():
    print("Failed to read video")
    exit()

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_dim = (width, height)
cv.namedWindow('Video', cv.WINDOW_NORMAL)
cv.resizeWindow('Video', width // 4, height // 4)

slam = Slam(width, height)
renderer = Renderer3D(pov_=90, cam_distance=2000)

skip_frame = 2
matches = None

frame_pixels = None
last_frame_pixels = None

while True:
    ret, frame_pixels = video.read()
    assert ret != None, "Failed to read video"
    frame_pixels = cv.resize(frame_pixels, video_dim, interpolation = cv.INTER_AREA)
    if last_frame_pixels is not None and not (frame_pixels==last_frame_pixels).all(): 
        slam.update_frame_pixels(current_frame_pixels=frame_pixels,
                                 last_frame_pixels=last_frame_pixels)
        matches, frame_pixels = slam.get_vision_matches(frame_pixels)
        if matches is not None:
            points = slam.triangulate(matches)
            if points is not None:
                renderer.render3dSpace(points, slam.get_camera_poses())
    last_frame_pixels = frame_pixels.copy()
    cv.imshow('Video', frame_pixels)
    renderer.render()
    key = cv.waitKey(25)
    if key == ord('q'):
        break
video.release()
cv.destroyAllWindows()
