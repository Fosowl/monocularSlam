# implementation of Simultaneous Localization and Mapping (SLAM)

import numpy as np
import math
import cv2 as cv

# rappel matrice:
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Vision():
    def __init__(self, video_dim) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.twin_points = None
        self.last_frame = None
        self.calibration_frames = []
        self.cx = video_dim[0] / 2
        self.cy = video_dim[1] / 2
        self.K = None # camera intrinsics (focal length, principal point)
        self.E = None # essental matrix
        self.focal = 1.0
        fx = self.focal
        fy = self.focal
        self.K = np.array([[fx, 0, self.cx],
                           [0, fy, self.cy],
                           [0, 0, 1]])
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))

    def estimate_pose(self, twin_points):
        c1 = []
        c2 = []
        for pt1, pt2 in twin_points:
            c1.append(pt1)
            c2.append(pt2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        focal = 1.0
        pp = (self.cx, self.cy) # principal point
        self.E, _ = cv.findEssentialMat(c1, c2, focal, pp, cv.RANSAC, 0.999, 1)
        _, R, t, _ = cv.recoverPose(self.E, c1, c2, self.K, pp)
        self.pose['R'] = R
        self.pose['t'] = t
        return self.E, self.pose

    def find_matching_points(self, frame):
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=2500, qualityLevel=0.01, minDistance=5)
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
        print(len(self.twin_points))
        for pt1, pt2 in self.twin_points:
            # current frame
            cv.circle(frame, (int(pt1[0]), int(pt1[1])), color=(0, 255, 0), radius=5)
            # previous frame
            cv.circle(frame, (int(pt2[0]), int(pt2[1])), color=(0, 0, 255), radius=5)
        E, pose = self.estimate_pose(self.twin_points)
        print("E", E)
        print("pose", pose)

class Slam():
    def __init__(self, width, height) -> None:
        self.vision = Vision(video_dim=(width, height))

    def view_points(self, frame):
        _ = self.vision.find_matching_points(frame)
        self.vision.view_interest_points(frame)