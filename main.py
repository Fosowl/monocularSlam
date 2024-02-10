#!/usr/bin python3

import cv2 as cv
from sources.slam import Frame
from sources.slam import Slam
from sources.render import Renderer3D

video = cv.VideoCapture('./videos/forest_road.mp4')
if not video.isOpened():
    print("Failed to read video")
    exit()

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_dim = (width, height)
cv.namedWindow('Video', cv.WINDOW_NORMAL)
cv.resizeWindow('Video', width // 2, height // 2)

slam = Slam(width, height)
renderer = Renderer3D()

skip_frame = 3
matches = None

frame_pixels = None
last_frame_pixels = None

while True:
    ret, frame_pixels = video.read()
    assert ret != None, "Failed to read video"
    frame_pixels = cv.resize(frame_pixels, video_dim, interpolation = cv.INTER_AREA)
    slam.update_frame_pixels(current_frame_pixels=frame_pixels,
                             last_frame_pixels=last_frame_pixels)
    matches, frame_pixels = slam.get_vision_matches(frame_pixels)
    if skip_frame == 0:
        last_frame_pixels = frame_pixels
        skip_frame = 3
    skip_frame -= 1
    if matches is not None:
        points, centroid = slam.triangulate(matches)
        #renderer.position_camera(position=(centroid[0], centroid[1], centroid[2]))
        renderer.render3dSpace(points)
    cv.imshow('Video', frame_pixels)
    renderer.render()
    key = cv.waitKey(25)
    if key == ord('q'):
        break
video.release()
cv.destroyAllWindows()
