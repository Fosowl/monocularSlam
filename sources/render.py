"""
Rendu 3D
"""

import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *
import math

class Camera:
    def __init__(self, fov=45, cam_distance_ = 100, position_=(0.0, 0.0, 0.0)) -> None:
        self.position = position_ 
        self._projection_matrix = None
        self._modelview_matrix = None
        self.orbital_radius = cam_distance_ # orbital_radius from the camera to the origin
        self.polar = np.deg2rad(0) # rads
        self.azimuth = np.deg2rad(0) # rads
        self.setup(fov_y=fov, aspect_ratio=1.0, near=10, far=50000.0)
        self.update()

    def set_origin(self, position=(0.0, 0.0, 0.0)):
        self.position = position

    def setup(self, fov_y=45, aspect_ratio=1.0, near=0.1, far=1000.0):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-aspect_ratio * math.tan(np.deg2rad(fov_y) / 2),
                  aspect_ratio * math.tan(np.deg2rad(fov_y) / 2),
                  -math.tan(np.deg2rad(fov_y) / 2),
                  math.tan(np.deg2rad(fov_y) / 2),
                  near,
                  far)
        self._projection_matrix = glGetFloatv(GL_PROJECTION_MATRIX)
        self._modelview_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
        print("Projection matrix:")
        print(self._projection_matrix)

    @property
    def get_projection_matrix(self):
        return self._projection_matrix
    
    @property
    def get_modelview_matrix(self):
        return self._modelview_matrix
    
    @property
    def get_position(self):
        return self.position

    def rotate_azimuth(self, degree=0.0):
        radians = np.deg2rad(degree)
        self.azimuth += radians
        circle = 2.0 * math.pi
        self.azimuth = self.azimuth % circle
        #if self.azimuth < 0:
        #    self.azimuth += circle

    def rotate_polar(self, degree=0.0):
        radians = np.deg2rad(degree)
        self.polar += radians
        polar_cap = math.pi / 2.0 - 0.01
        #if self.polar > polar_cap:
        #    self.polar = polar_cap - 0.01
        #elif self.polar < -polar_cap:
        #    self.polar = -polar_cap + 0.01
    
    def zoom(self, by=0.0):
        self.orbital_radius += by
        if self.orbital_radius < 10:
            self.orbital_radius = 10

    def update(self, rotation_center=(0, 0, 0)):
        x = self.orbital_radius * np.cos(self.polar) * np.cos(self.azimuth)
        y = self.orbital_radius * np.sin(self.polar)
        z = self.orbital_radius * np.cos(self.polar) * np.sin(self.azimuth)
        self.position = (x,
                         y,
                         z)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.position, # from
                  rotation_center[0], rotation_center[1], rotation_center[2], # look at origin
                  0, 1, 0) # axis
    

class Renderer3D:
    def __init__(self, pov_=90.0, cam_distance=100) -> None:
        self.window = (800, 600)
        self.fov = pov_
        pygame.display.set_mode(self.window, DOUBLEBUF | OPENGL)
        self.camera = Camera(fov=self.fov, cam_distance_=cam_distance)
        self.pause = False
        self.saved_points = None

    def handle_inputs(self):
        rotation_speed = 15
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.camera.rotate_azimuth(rotation_speed)
            if keys[pygame.K_RIGHT]:
                self.camera.rotate_azimuth(-rotation_speed)
            if keys[pygame.K_UP]:
                self.camera.rotate_polar(rotation_speed)
            if keys[pygame.K_DOWN]:
                self.camera.rotate_polar(-rotation_speed)
            if keys[pygame.K_ESCAPE] and event.type == pygame.KEYDOWN:
                self.pause = not self.pause
                print("Paused" if self.pause else "Unpaused")

    def draw_cube(self):
        vertices = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                    (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
    
    def draw_lines(self, start, end, color=(0.0, 1.0, 0.0)):
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(*start)
        glVertex3f(*end)
        glEnd()

    def render_axis(self):
        self.draw_lines(start=(0, 0, 0), end=(10, 0, 0), color=(1.0, 0.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 10, 0), color=(0.0, 1.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 0, 10), color=(0.0, 0.0, 1.0))
    
    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)

    def draw_points(self, points, slam_proj_matrix, position, color=(0.0, 1.0, 0.0)):
        assert len(points) > 0, "No points to draw"
        """
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX).astype('d')
        proj = slam_proj_matrix.astype('d')
        proj = glGetFloatv(GL_PROJECTION_MATRIX).astype('d') 
        """
        glBegin(GL_POINTS)
        max_z = max(points, key=lambda x: np.abs(x[2]))[2]
        max_y = max(points, key=lambda x: np.abs(x[1]))[1]
        max_x = max(points, key=lambda x: np.abs(x[0]))[0]
        print("Max x: ", max_x, "Max y: ", max_y, "Max z: ", max_z)
        for i, point in enumerate(points):
            color = (1, 0.3, 0)
            glColor3f(*color)
            # no need to project since slam already use w component of vector to project points ?
            # https://learnopengl.com/Getting-started/Coordinate-Systems
            #point_proj = gluProject(point[0], point[1], point[2], modelview, proj, viewport)
            point_corrected = (point[0] + position[0],
                               point[1] + position[1],
                               point[2] + position[2])
            glVertex3f(*point_corrected)
        glEnd()

    def draw_trajectory(self, camera_poses, color=(0.5, 0.5, 1.0)):
        T_total = np.zeros((3, 1))
        R_total = np.eye(3)
        glColor3f(*color)
        glBegin(GL_LINE_STRIP)
        for pose in camera_poses:
            T_total += R_total @ pose['t']
            R_total = R_total @ pose['R']
            glVertex3f(*T_total)
        print(f"Last position:\n{T_total}")
        glEnd()
        return T_total


    @property
    def is_paused(self):
        return self.pause

    def render3dSpace(self, points3D, slam_proj_matrix, camera_poses=None):
        if points3D is None:
            print("No points to render")
            return
        glClearColor(0, 0, 0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.render_axis()
        print(f"Rendering {len(points3D)} points")
        position = self.draw_trajectory(camera_poses)
        self.draw_points(points3D, slam_proj_matrix, position)
        self.draw_cube()
        assert camera_poses != None, "No camera poses given"
        # ugly but avoid 2 loops
        self.handle_inputs()
        self.camera.update()
