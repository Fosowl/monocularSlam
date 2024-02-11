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
    def __init__(self, fov=60, orientation_=(0.0, 0.0, 0.0), position_=(0.0, 0.0, 0.0)) -> None:
        self.position = position_ 
        self.projection_matrix = None
        self.modelview_matrix = None
        self.orbital_radius = 40 # orbital_radius from the camera to the origin
        self.polar = 0 # rads
        self.azimuth = 0 # rads
        self.setup(fov_y=fov, aspect_ratio=1.0, near=0.1, far=25000.0)
        self.update()

    def set_origin(self, position=(0.0, 0.0, 0.0)):
        self.position = position

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

    @property
    def get_projection_matrix(self):
        return self.projection_matrix
    
    @property
    def get_modelview_matrix(self):
        return self.modelview_matrix

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

    def handle_mouse(self):
        mouse_dx, mouse_dy = 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
                mouse_dx, mouse_dy = pygame.mouse.get_rel()
        self.rotate_azimuth(-mouse_dx*0.5)
        self.rotate_polar(mouse_dy*0.5)
        print("Cam position: ", self.position)
        print("Cam orbital_radius: ", self.orbital_radius)
        print("Cam polar: ", np.rad2deg(self.polar))
        print("Cam azimuth: ", np.rad2deg(self.azimuth))

    def update(self, rotation_center=(0, 0, 0)):
        self.handle_mouse()
        x = self.orbital_radius * np.cos(self.polar) * np.cos(self.azimuth) + rotation_center[0]
        y = self.orbital_radius * np.sin(self.polar) + rotation_center[1]
        z = self.orbital_radius * np.cos(self.polar) * np.sin(self.azimuth) + rotation_center[2]
        self.position = (x,
                         y,
                         z)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(*self.position)  # Position the camera
        gluLookAt(*rotation_center, 0, 0, 0, 0, 1, 0)
        self.modelview_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32)
        glMultMatrixf(self.modelview_matrix.tobytes())

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
        self.draw_lines(start=(0, 0, 0), end=(1000, 0, 0), color=(1.0, 0.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 1000, 0), color=(0.0, 1.0, 0.0))
        self.draw_lines(start=(0, 0, 0), end=(0, 0, 1000), color=(0.0, 0.0, 1.0))

    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(*color)
        glBegin(GL_POINTS)
        glVertex3f(*point)
        glEnd()

    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)

    def render3dSpace(self, points3D, centroid):
        if points3D is None:
            return

        glClear(GL_COLOR_BUFFER_BIT)
        self.render_axis()
        for i, point in enumerate(points3D):
            r = (i + 1) / len(points3D)
            color = (r, 255, 0.0)
            point = point - centroid
            self.draw_point(point, color)
        self.draw_cube()
        self.camera.update()

def main():
    renderer = Renderer3D()
    while True:
        renderer.render3dSpace(np.array([]), np.array([0, 0, 0]))
        renderer.render()

if __name__ == "__main__":
    main()
