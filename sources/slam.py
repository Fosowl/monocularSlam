"""
Implementation SLAM
"""

import numpy as np
import cv2 as cv
from typing import Tuple
import math
import copy

# rappel matrice:
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Frame():
    def __init__(self) -> None:
        self.pixels = None
        self.kps = None
        self.des = None
        self.E = None
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))
    
    def assign_frame_pixels(self, pixels: np.ndarray):
        self.pixels = pixels
    
    def __str__(self) -> str:
        return f"Data: {len(self.pixels)}, kps: {self.kps}, des: {self.des}, E: {self.E}, pose: {self.pose}"

class Vision():
    def __init__(self, video_dim: Tuple[int, int], _focal) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.focal = _focal
        self.cx = video_dim[0] // 2
        self.cy = video_dim[1] // 2
        self.K = np.array([[self.focal, 0, self.cx], # camera intrisics
                           [0, self.focal, self.cy],
                           [0, 0, 1]])
        print("Camera intrisics matrix:")
        print(self.K)
        self.current_frame = Frame()
        self.last_frame = Frame()
        self.matches = None

    def get_camera_pose(self, matches: Tuple[Tuple[float, float], Tuple[float, float]]):
        assert matches != None, "No matches given"
        c1 = []
        c2 = []
        for pt1, pt2 in matches:
            c1.append(pt1)
            c2.append(pt2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        pp = (self.cx, self.cy) # principal point
        E, _ = cv.findEssentialMat(c1, c2, self.focal, pp, cv.RANSAC, 0.999, 1)
        _, R, t, _ = cv.recoverPose(E, c1, c2, self.K, pp)
        pose = dict()
        pose['R'] = R
        pose['t'] = t
        self.current_frame.E = E
        self.current_frame.pose = pose

    def distance_between_points(self, pt1: float, pt2: float):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def find_matching_points(self, current_frame: Frame, last_frame: Frame):
        assert current_frame.pixels is not None, "No frame passed"
        match = np.mean(current_frame.pixels, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=150, qualityLevel=0.01, minDistance=5)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(current_frame.pixels, kps)
        self.current_frame.kps = kps
        self.current_frame.des = des
        if self.last_frame.kps is None or self.last_frame.des is None:
            return None
        self.matches = []
        for m in self.matcher.match(des, self.last_frame.des):
            kp1 = self.current_frame.kps[m.queryIdx].pt # (float, float)
            kp2 = self.last_frame.kps[m.trainIdx].pt # (float, float)
            self.matches.append((kp1, kp2))
        return self.matches

    def view_interest_points(self, frame: Frame, matches: Tuple[Tuple[float, float], Tuple[float, float]]):
        assert matches != None, "No matches passed"
        assert frame.pixels is not None, "No frame passed"
        for pt1, pt2 in matches:
            # current frame
            cv.circle(frame.pixels, (int(pt1[0]), int(pt1[1])), color=(0, 255, 0), radius=6)
            # previous frame
            cv.circle(frame.pixels, (int(pt2[0]), int(pt2[1])), color=(0, 0, 255), radius=10)
            # line
            cv.line(frame.pixels, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=3)
        return frame.pixels

class Slam():
    def __init__(self, width: int, height: int) -> None:
        self.vision = Vision(video_dim=(width, height), _focal=525)

    def update_frame_pixels(self, current_frame_pixels: np.ndarray, last_frame_pixels: np.ndarray):
        if last_frame_pixels is not None:
            self.vision.last_frame = self.vision.current_frame
            self.vision.last_frame.assign_frame_pixels(last_frame_pixels)
        self.vision.current_frame.assign_frame_pixels(current_frame_pixels)
    
    def get_vision_matches(self, render_frame):
        assert self.vision.current_frame.pixels is not None, "No frame passed"
        matches = self.vision.find_matching_points(self.vision.current_frame,
                                                   self.vision.last_frame)
        if matches is not None:
            render_frame = self.vision.view_interest_points(self.vision.current_frame, matches)
            self.vision.get_camera_pose(matches)
            return matches, render_frame
        print("No matches found")
        return None, render_frame

    def triangulate(self, matches: Tuple[Tuple[float, float], Tuple[float, float]]):
        assert matches != None, "No matches passed"
        assert len(matches) > 0, "No matches passed"
        assert self.vision.current_frame.E is not None, "current essential matrix is None"
        assert self.vision.current_frame.pose is not None, "current pose is None"
        assert self.vision.last_frame.E is not None, "Last frame essential matrix is None"
        assert self.vision.last_frame.pose is not None, "Last frame pose is None"
        projection_matrix = np.hstack((self.vision.current_frame.pose['R'],
                                       self.vision.current_frame.pose['t']))
        past_projection_matrix = np.hstack((self.vision.last_frame.pose['R'],
                                            self.vision.last_frame.pose['t']))
        projPoints1 = []
        projPoints2 = []
        for kp1, kp2 in matches:
            projPoints1.append([kp1[0], kp1[1]])
            projPoints2.append([kp2[0], kp2[1]])
        projPoints1 = np.array(projPoints1).T  # (2, N)
        projPoints2 = np.array(projPoints2).T
        points4D = cv.triangulatePoints(past_projection_matrix, projection_matrix, projPoints1, projPoints2)
        scales = points4D[3]
        points3D = (points4D[:3] / scales).T
        points3D *= 100 # scaling up for 3D rendering
        centroid = sum([v for v in points3D]) / len(points3D)
        print(f"3D centroid: {centroid}")
        return points3D, centroid
    