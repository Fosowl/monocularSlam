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
    def __init__(self, fov=60, orientation_=(0.0, 0.0, 0.0), position_=(0.0, 0.0, -4.0)) -> None:
        self.orientation = orientation_
        self.position = position_ 
        self.projection_matrix = None
        self.modelview_matrix = None
        self.forward = (0.0, 0.0, 1.0)
        self.setup()
        self.update()

    def rotate_camera(self, angle_x=0.0, angle_y=0.0, angle_z=0.0):
        self.orientation = (angle_x, angle_y, angle_z)
    
    def position_camera(self, position=(0.0, 0.0, 0.0)):
        self.position = position

    @property
    def get_projection_matrix(self):
        return self.projection_matrix
    
    @property
    def get_modelview_matrix(self):
        return self.modelview_matrix

    def update_forward_vector(self):
        pitch, yaw, _ = map(np.deg2rad, self.orientation)  # Convert to radians
        self.forward = (
            np.cos(pitch) * np.cos(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.sin(yaw)
        )

    def update(self, rotation_center=(0, 0, 0)):
        self.handle_mouse()
        self.update_forward_vector()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(*self.position)  # Position the camera
        angle_x, angle_y, angle_z = self.orientation
        glRotatef(angle_x, 1, 0, 0)  # Rotate around x-axis
        glRotatef(angle_y, 0, 1, 0)  # Rotate around y-axis
        glRotatef(angle_z, 0, 0, 1)  # Rotate around y-axis
        self.modelview_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32)
        glMultMatrixf(self.modelview_matrix.tobytes())

    def setup(self, fov_y=60, aspect_ratio=1.0, near=0.1, far=1500.0):
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
    
    def handle_mouse(self):
        mouse_dx, mouse_dy = 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEMOTION and event.buttons[0] == 1:
                mouse_dx, mouse_dy = event.rel
            if not event.type == pygame.MOUSEBUTTONDOWN:
                continue
            if event.button == 4:
                self.position = [pos + 0.5 * forward for pos, forward in zip(self.position, self.forward)]
            elif event.button == 5:
                self.position = [pos - 0.5 * forward for pos, forward in zip(self.position, self.forward)]
            
        self.orientation = (self.orientation[0] + mouse_dx * 1.0,
                            self.orientation[1] + mouse_dy * 1.0,
                            self.orientation[2])
        print("Cam orientation: ", self.orientation)
    
class Renderer3D:
    def __init__(self) -> None:
        self.window = (800, 600)
        self.fov = 60
        pygame.display.set_mode(self.window, DOUBLEBUF | OPENGL)
        gluPerspective(self.fov, (self.window[0] / self.window[1]), 1.1, 5000.0) # TODO maybe change
        self.camera = Camera(fov=self.fov)

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
        glMatrixMode(GL_MODELVIEW)
        self.draw_lines(start=(0, 0, 0), end=(100, 0, 0), color=(1.0, 0.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 100, 0), color=(0.0, 1.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 0, 1000), color=(0.0, 0.0, 1.0))

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
    
    def render3dSpace(self, points3D, centroid):
        if points3D is None:
            return
        # Convert points to OpenGL coordinate system (right handed)
        points3D = self.convert_points_to_opengl(points3D)
        transformation_matrix = self.create_transformation_matrix(centroid)
        points3D = np.dot(transformation_matrix, np.concatenate([points3D, np.ones((points3D.shape[0], 1))], axis=1).T).T[:, :3]

        glClear(GL_COLOR_BUFFER_BIT)
        print("Rendering {} points".format(len(points3D)))
        for i, point in enumerate(points3D):
            r = (i + 1) / len(points3D)
            color = (r, 255, 0.0)
            point = point - centroid
            self.draw_point(point, color)
        self.camera.update()
        self.render_axis()
        self.draw_cube()