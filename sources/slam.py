"""
SLAM
"""

import numpy as np
import cv2 as cv
from typing import Tuple, List
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
    
    def copy(self, that_frame, pixels):
        self.pixels = pixels.copy()
        self.kps = that_frame.kps
        self.des = that_frame.des
    
    def __str__(self) -> str:
        return f"Data: {len(self.pixels)}, kps: {self.kps}, des: {self.des}, E: {self.E}, pose: {self.pose}"

class Vision():
    def __init__(self, video_dim: Tuple[int, int], _focal: float) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.feats = None
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
        self.camera_poses = []
        self.T_total = np.zeros((3, 1))
        self.R_total = np.eye(3)
    
    def get_camera_poses(self):
        return self.camera_poses
    
    def camera_pose_to_opengl(self, T_total, R_total):
        pose = dict()
        pose['R'] = R_total
        pose['t'] = T_total
        corrected_pose = copy.deepcopy(pose)
        corrected_pose['t'][2] *= -1
        corrected_pose['t'][1] *= -1
        return corrected_pose

    def get_camera_pose(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
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
        self.camera_poses.append(self.camera_pose_to_opengl(self.T_total, self.R_total))
        self.T_total += self.R_total @ pose['t']
        self.R_total = self.R_total @ pose['R']
    
    def get_pose_cumulation(self):
        return self.R_total, self.T_total

    def distance_between_points(self, pt1: float, pt2: float):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def find_matching_points(self, current_frame: Frame):
        assert current_frame.pixels is not None, "No frame passed"
        match = np.mean(current_frame.pixels, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=7)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(current_frame.pixels, kps)
        self.feats = feats
        self.current_frame.kps = kps
        self.current_frame.des = des
        if self.last_frame.kps is None or self.last_frame.des is None:
            return None
        self.matches = []
        for m in self.matcher.match(des, self.last_frame.des):
            kp1 = self.current_frame.kps[m.queryIdx].pt # (float, float)
            kp2 = self.last_frame.kps[m.trainIdx].pt # (float, float)
            if self.distance_between_points(kp1, kp2) > 25:
                continue
            if kp1 != kp2:
                self.matches.append((kp1, kp2))
        assert len(self.matches) > 0, "No matches found"
        return self.matches

    def view_interest_points(self, frame: Frame, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "No matches passed"
        assert frame.pixels is not None, "No frame passed"

        for i, (pt1, pt2) in enumerate(matches):
            assert pt1 != pt2, "Points are the same"
            # current frame
            cv.circle(frame.pixels, (int(pt1[0]), int(pt1[1])), color=(0, 255, 255), radius=4)
            # previous frame
            cv.circle(frame.pixels, (int(pt2[0]), int(pt2[1])), color=(255, 0, 255), radius=3)
            # line
            cv.line(frame.pixels, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=1)
        return frame.pixels

class Slam():
    def __init__(self, width: int, height: int) -> None:
        self.vision = Vision(video_dim=(width, height), _focal=525)
        self._projection_matrix = None
        self._past_projection_matrix = None
        self.E_buffer = None
        self.pose_buffer = None
        self.points_centroid = None
        self.points3Dcumulative = []
    
    @property
    def projection_matrix(self):
        return self._projection_matrix
    
    @property
    def past_projection_matrix(self):
        return self._past_projection_matrix
    
    def get_camera_poses(self):
        return self.vision.get_camera_poses()

    def update_frame_pixels(self, current_frame_pixels: np.ndarray, last_frame_pixels: np.ndarray):
        assert current_frame_pixels is not None, "No frame passed"
        assert last_frame_pixels is not None, "No last frame passed"
        assert (current_frame_pixels==last_frame_pixels).all() == False, "Frames are the same"
        if last_frame_pixels is not None:
            self.vision.last_frame.copy(that_frame=self.vision.current_frame, pixels=last_frame_pixels)
        self.vision.current_frame.pixels = current_frame_pixels.copy()
    
    def get_vision_matches(self, render_frame):
        assert render_frame is not None, "No frame for rendering"
        assert self.vision.current_frame.pixels is not None, "No frame passed"
        assert self.vision.last_frame.pixels is not None, "No last frame"
        assert (self.vision.current_frame.pixels==self.vision.last_frame.pixels).all() == False, "Frames are the same"
        matches = self.vision.find_matching_points(self.vision.current_frame)
        if matches is not None:
            render_frame = self.vision.view_interest_points(self.vision.current_frame, matches)
            self.vision.get_camera_pose(matches)
            return matches, render_frame
        print("No matches found")
        return None, render_frame

    def hand_rule_change(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] > 4, "Points4D not 4xN"
        # change coordinate opencv to openGL
        points3D[:, 1] *= -1
        points3D[:, 2] *= -1
        return points3D
    
    def transform_points_3D_openGL(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] == 3, "Points3D not 3D"
        R, t = self.vision.get_pose_cumulation()
        R_inv = R.T
        t_inv = -R.T @ t
        return np.dot(points3D.T, R_inv) + t_inv.T
    
    # useless
    def project_points(self, points3D):
        inv = np.linalg.inv(self.vision.K)
        points3D_projected = np.dot(inv, points3D)
        return points3D_projected
    
    def triangulate(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "matches is None"
        assert len(matches) > 0, "No matches passed"
        assert self.vision.current_frame.E is not None, "current essential matrix is None"
        assert self.vision.current_frame.pose is not None, "current pose is None"
        if self.vision.last_frame.E is None:
            self.vision.last_frame.E = self.vision.current_frame.E
            self.vision.last_frame.pose = self.vision.current_frame.pose
        self._projection_matrix = np.hstack((self.vision.current_frame.pose['R'],
                                       self.vision.current_frame.pose['t']))
        self._past_projection_matrix = np.hstack((self.vision.last_frame.pose['R'],
                                            self.vision.last_frame.pose['t']))
        projPoints1 = []
        projPoints2 = []
        for kp1, kp2 in matches:
            projPoints1.append([kp1[0], kp1[1]])
            projPoints2.append([kp2[0], kp2[1]])
        projPoints1 = np.array(projPoints1).T  # (2, N)
        projPoints2 = np.array(projPoints2).T
        K = self.vision.K
        projMat1 = K @ cv.hconcat([self.vision.current_frame.pose['R'], self.vision.current_frame.pose['t']]) # Cam1 is the origin
        projMat2 = K @ cv.hconcat([self.vision.last_frame.pose['R'], self.vision.last_frame.pose['t']]) # R, T from stereoCalibrate
        points1u = cv.undistortPoints(projPoints1, K, 1, None, K)
        points2u = cv.undistortPoints(projPoints2, K, 1, None, K)
        points4D = cv.triangulatePoints(projMat1, projMat2, points1u, points2u)
        # 3d
        points3D = (points4D[:3] / points4D[3]) # normalize and go 3d
        goods = np.abs(points4D[3, :]) > 0.005 # check w good
        goods &= points3D[1, :] > 0 # check point are not underground
        goods &= points3D[2, :] > 0 # check points are in front of camera
        goods &= points3D[2, :] < 500 # check points are not too far
        goods &= np.abs(points3D[0, :]) < 500 # check points are not too far
        if len(goods[goods == True]) < 25:
            print("error: not enough points")
            return None
        points3D = points3D[:, goods]
        points3D = self.transform_points_3D_openGL(points3D)
        self.points_centroid = sum([v for v in points3D]) / len(points3D)
        print("SLAM: points centroid: ", self.points_centroid)
        print(f"SLAM: estimate position: {self.vision.T_total[0]}, {self.vision.T_total[1]}, {self.vision.T_total[2]}")
        self.vision.last_frame.E = self.vision.current_frame.E
        self.vision.last_frame.pose = self.vision.current_frame.pose
        point_info = (points3D, self.points_centroid)
        print(f"info: {point_info}")
        self.points3Dcumulative.append(point_info)
        return self.points3Dcumulative
    
