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
    def __init__(self, fov=45, cam_distance_ = 50, position_=(0.0, 0.0, 0.0)) -> None:
        self.position = position_ 
        self.projection_matrix = None
        self.modelview_matrix = None
        self.orbital_radius = cam_distance_ # orbital_radius from the camera to the origin
        self.polar = np.deg2rad(0) # rads
        self.azimuth = np.deg2rad(0) # rads
        self.setup(fov_y=fov, aspect_ratio=1.0, near=10, far=1000.0)
        self.update()

    def set_origin(self, position=(0.0, 0.0, 0.0)):
        self.position = position

    def setup(self, fov_y=45, aspect_ratio=1.0, near=0.1, far=5000.0):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-aspect_ratio * math.tan(np.deg2rad(fov_y) / 2),
                  aspect_ratio * math.tan(np.deg2rad(fov_y) / 2),
                  -math.tan(np.deg2rad(fov_y) / 2),
                  math.tan(np.deg2rad(fov_y) / 2),
                  near,
                  far)
        self.projection_matrix = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32)
        print("Projection matrix:")
        print(self.projection_matrix)

    @property
    def get_projection_matrix(self):
        return self.projection_matrix
    
    @property
    def get_modelview_matrix(self):
        return self.modelview_matrix
    
    @property
    def get_position(self):
        return self.position

    def rotate_azimuth(self, degree=0.0):
        radians = np.deg2rad(degree)
        self.azimuth += radians
        circle = 2.0 * math.pi
        self.azimuth = self.azimuth % circle
        if self.azimuth < 0:
            self.azimuth += circle

    def rotate_polar(self, degree=0.0):
        radians = np.deg2rad(degree)
        self.polar += radians
        polar_cap = math.pi / 2.0 - 0.01
        if self.polar > polar_cap:
            self.polar = polar_cap
        elif self.polar < -polar_cap:
            self.polar = -polar_cap
    
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
        gluLookAt(*self.position, 0, 0, 0, 0, 1, 0)
    
class Renderer3D:
    def __init__(self, pov_=45.0, cam_distance=50) -> None:
        self.window = (800, 600)
        self.fov = pov_
        pygame.display.set_mode(self.window, DOUBLEBUF | OPENGL)
        self.camera = Camera(fov=self.fov, cam_distance_=cam_distance)
        self.pause = False
        self.saved_points = None

    def handle_inputs(self):
        rotation_speed = 10
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
    
    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(*color)
        glBegin(GL_POINTS)
        glVertex3f(*point)
        glEnd()

    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)

    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(*color)
        glBegin(GL_POINTS)
        glVertex3f(*point)
        glEnd()

    def convert_points_to_opengl(self, points3D):
        points3D[:, 1] *= -1
        points3D[:, 2] *= -1
        return points3D

    def create_transformation_matrix(self, centroid):
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, 3] = -centroid
        return transformation_matrix

    def render3dSpace(self, points3D_slam, centroid):
        if points3D_slam is None:
            return
        if not self.pause:
            self.saved_points = self.convert_points_to_opengl(points3D_slam)
        glClearColor(0, 0, 0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.render_axis()
        for i, point in enumerate(self.saved_points):
            r = (i + 1) / len(self.saved_points)
            point -= centroid
            color = (r, 255, 0.0)
            self.draw_point(point, color)
        self.draw_cube()
        self.handle_inputs()
        self.camera.update()