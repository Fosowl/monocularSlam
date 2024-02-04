# implementation of Simultaneous Localization and Mapping (SLAM)

import numpy as np
import math
import cv2 as cv

class Vision():
    def __init__(self) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.twin_points = None
        self.last_frame = None

    def find_matching_points(self, frame):
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame, kps)
        self.last_frame = {'kps': kps, 'des': des}
        if self.last_frame is None:
            return None
        matches = self.matcher.match(des, self.last_frame['des'])
        self.twin_points = []
        for m in matches:
            kp1 = kps[m.queryIdx].pt
            kp2 = self.last_frame['kps'][m.trainIdx].pt
            self.twin_points.append((kp1, kp2))
        return self.twin_points

    def view_interest_points(self, frame):
        if self.twin_points is None:
            print("no matches")
            return
        for pt1, pt2 in self.twin_points:
            # current frame
            cv.circle(frame, (int(pt1[0]), int(pt1[1])), color=(0, 255, 0), radius=4)
            # previous frame
            cv.circle(frame, (int(pt2[0]), int(pt2[1])), color=(0, 0, 255), radius=4)

class Slam():
    def __init__(self) -> None:
        self.vision = Vision()

    def view_points(self, frame):
        _ = self.vision.find_matching_points(frame)
        self.vision.view_interest_points(frame)