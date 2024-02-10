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

class Renderer3D:
    def __init__(self) -> None:
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        self.projection_matrix = None
        self.modelview_matrix = None
        self.setup_camera()
        self.position_camera()
    
    @property
    def get_projection_matrix(self):
        return self.projection_matrix
    
    @property
    def get_modelview_matrix(self):
        return self.modelview_matrix

    def position_camera(self, position=(0.0, 0.0, 10.0), angle_x=0.0, angle_y=0.0):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(*position)  # Position the camera
        glRotatef(angle_x, 1.0, 0.0, 0.0)  # Rotate around x-axis
        glRotatef(angle_y, 0.0, -1.0, 0.0)  # Rotate around y-axis
        self.modelview_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32)
        glMultMatrixf(self.modelview_matrix.tobytes())
        print("Modelview matrix:")
        print(self.modelview_matrix)

    def setup_camera(self, fov_y=90.0, aspect_ratio=1.0, near=0.1, far=1500.0):
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
    
    def draw_lines(self, start, end, color=(0.0, 1.0, 0.0)):
        glColor3f(color)
        glVertex3f(start)
        glVertex3f(end)

    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)

    #def draw_point(self, point, color=(0.0, 1.0, 0.0)):
    #    glColor3f(color[0], color[1], color[2])
    #    glVertex3f(point[0], point[1], point[2])

    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_POINTS)
        glVertex3f(point[0], point[1], point[2])
        glEnd()
    
    def render3dSpace(self, points):
        if points is None:
            return
        glClear(GL_COLOR_BUFFER_BIT)
        print("Rendering {} points".format(len(points)))
        for point in points:
            self.draw_point(point)