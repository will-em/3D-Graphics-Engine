import pygame
from pygame import gfxdraw
import numpy as np
import copy
import time
import operator
import math
import os 
import sys

transDist = 10
rotSpeed = 0.0
speed = 0.2
rot_speed = 0.05

lookDir = np.array([0, 0, 1])
lookDir = lookDir/np.linalg.norm(lookDir)
sensitivity = 0.005

height = 800
width = 800

#Projection Matrix
fNear = 0.1
fFar = 1000.0
fFov = 90.0
fAspectRatio = width/height
fFovRad = 1.0/np.tan(np.deg2rad(fFov/2))

proj_mat = np.array([
    [fFovRad / fAspectRatio, 0, 0, 0],
    [0, fFovRad, 0, 0],
    [0, 0, fFar / (fFar - fNear), 1],
    [0, 0, (-fFar * fNear) / (fFar - fNear), 0]
])

color1 = (255, 0, 0)
color2 = (0, 255, 0)
color3 = (0, 0, 255)
color4 = (255, 0, 255)
color5 = (255, 255, 51)
color6 = (102, 255, 255)
colors = [color1, color1, color2, color2, color3, color3, color4, color4, color5, color5, color6, color6]

class Triangle:
    def __init__(self, p):
        self.p = p

    def __str__(self):
        return str(self.points[0])

    def getZmean(self):
        return (self.p[0][2]+self.p[1][2]+self.p[2][2])/3

    #def __lt__(self, other):
    #    selfzMean = (self.p[0][2]+self.p[1][2]+self.p[2][2])/3
    #    otherzMean = (other.p[0][2]+other.p[1][2]+other.p[2][2])/3
    #    return otherzMean>selfzMean

class Mesh:
    def __init__(self):
        pass

def affine_transform(mat, vec):
    aug_vec = np.append(vec, 1.0)

    res = mat.T @ aug_vec
    w = res[-1]
    if w != 0:
        res[:] = res / w

    return res[:-1]

def loadMesh(path):
    verts = [] #To store the pool of vertices
    tris = [] #Triangles
    with open(path, "r") as file:
        for row in file:
            subStr = row.split()
            if row[0]=="v":
                verts.append([float(subStr[1]), float(subStr[2]), float(subStr[3])])
            elif row[0]=="f":
                vert1 = verts[int(subStr[1])-1]
                vert2 = verts[int(subStr[2])-1]
                vert3 = verts[int(subStr[3])-1]
                tris.append(Triangle(np.array([vert1, vert2, vert3])))

    return tris

def rotMatX(theta):
    return np.array([
        [1, 0, 0, 0], 
        [0, np.cos(theta), -np.sin(theta), 0], 
        [0, np.sin(theta), np.cos(theta), 0], 
        [0, 0, 0, 1]
    ])

def rotMatY(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0], 
        [0, 1, 0, 0], 
        [-np.sin(theta), 0, np.cos(theta), 0], 
        [0, 0, 0, 1]
    ])

def rotMatZ(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0], 
        [np.sin(theta), np.cos(theta), 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]
    ])

def get_control_input(camera_rot):
    done = False
    delta_pos = np.zeros(3, dtype=np.float32)
    delta_rot = np.zeros(3, dtype=np.float32)
    look_dir = affine_transform(camera_rot, np.array([0.0, 0.0, 1.0]))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN :
            if event.key == pygame.K_LEFT:
                delta_pos[0] = -speed
            elif event.key == pygame.K_RIGHT:
                delta_pos[0] = speed
            if event.key == pygame.K_UP:
                delta_pos[1] = speed
            elif event.key == pygame.K_DOWN:
                delta_pos[1] = -speed
            if event.key == pygame.K_w:
                delta_pos = speed * look_dir
            elif event.key == pygame.K_s:
                delta_pos = -speed * look_dir

            if event.key == pygame.K_j:
                delta_rot[1] = rot_speed
            elif event.key == pygame.K_l:
                delta_rot[1] = -rot_speed

            if event.key == pygame.K_i:
                delta_rot[0] = rot_speed
            elif event.key == pygame.K_k:
                delta_rot[0] = -rot_speed

            if event.key == pygame.K_u:
                delta_rot[2] = rot_speed
            elif event.key == pygame.K_o:
                delta_rot[2] = -rot_speed

            if event.key == pygame.K_ESCAPE:
                done = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                delta_pos[0] = speed
            elif event.key == pygame.K_RIGHT:
                delta_pos[0] = -speed
            if event.key == pygame.K_UP:
                delta_pos[1] = -speed
            elif event.key == pygame.K_DOWN:
                delta_pos[1] = speed
            if event.key == pygame.K_w:
                delta_pos = -speed * look_dir
            elif event.key == pygame.K_s:
                delta_pos = speed * look_dir

            if event.key == pygame.K_j:
                delta_rot[1] = -rot_speed
            elif event.key == pygame.K_l:
                delta_rot[1] = rot_speed

            if event.key == pygame.K_i:
                delta_rot[0] = -rot_speed
            elif event.key == pygame.K_k:
                delta_rot[0] = rot_speed

            if event.key == pygame.K_u:
                delta_rot[2] = -rot_speed
            elif event.key == pygame.K_o:
                delta_rot[2] = rot_speed

    dMouse = pygame.mouse.get_rel()
    dMousex = dMouse[0]
    dMousey = dMouse[1]

    delta_rot[0] -= dMousey*sensitivity
    delta_rot[1] -= dMousex*sensitivity
                
    return done, delta_pos, delta_rot

def main():
    
    if len(sys.argv) != 2:
        raise ValueError("Must specify 3D object file")

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    done = False
    pygame.event.set_grab(True)
    pygame.mouse.set_pos = (width/2, height/2)
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()

    drawMesh = False
    line_width = 2
    meshCube = Mesh()
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, sys.argv[1])
    meshCube.tris = loadMesh(filename)

    theta = 0

    camera_pos = np.array([0.0, 0.0, -10.0]) # Initial camera position
    camera_rot = rotMatY(0).dot(rotMatX(0)) # Initial camera orientation
    camera_velocity = np.zeros(3, dtype=np.float32)
    camera_rot_speed = np.zeros(3, dtype=np.float32)

    while not done:

        done, delta_pos, delta_rot = get_control_input(camera_rot)

        camera_velocity += delta_pos
        camera_pos += camera_velocity

        camera_rot_speed += delta_rot
        camera_rot = camera_rot @ rotMatZ(camera_rot_speed[2]) @ rotMatY(camera_rot_speed[1]) @ rotMatX(camera_rot_speed[0])

        screen.fill((0, 0, 0))

        theta += rotSpeed
        rotzMat = rotMatZ(theta/2)
        rotxMat = rotMatX(theta)

        totMat = rotzMat @ rotxMat


        camera_mat = camera_rot.copy()
        camera_mat[3][:3] = camera_pos

        view_mat = np.linalg.inv(camera_mat)

        to_draw = []
        for k, tri in enumerate(meshCube.tris):
            trans_pts = []
            viewed_pts = []
            proj_pts = []
            for point in tri.p:
                point = point.copy()
                #Rotate
                point = affine_transform(totMat, point)

                #Translate
                point[2] += transDist
                trans_pts.append(point)

                #Worldspace to Viewspace
                point = affine_transform(view_mat, point)
                viewed_pts.append(point)

                #Project
                proj_pts.append(affine_transform(proj_mat, point))

            vec1 = [trans_pts[1][0]-trans_pts[0][0], trans_pts[1][1]-trans_pts[0][1], trans_pts[1][2]-trans_pts[0][2]]
            vec2 = [trans_pts[2][0]-trans_pts[0][0], trans_pts[2][1]-trans_pts[0][1], trans_pts[2][2]-trans_pts[0][2]]

            normal = np.cross(vec1, vec2, axis=0)
            normal = normal/(np.linalg.norm(normal)+1e-16)

            diff_vec = trans_pts[0] - camera_pos

            light_direction = np.array([0, 0, -1]) #Single Direction Light
            if np.dot(diff_vec, normal) < 0.0:
                light_direction = light_direction / np.linalg.norm(light_direction)
                light_dot = np.dot(light_direction, normal)
                if light_dot<0:
                    light_dot=0

                to_draw.append((proj_pts, light_dot))
        
        #Painter's algorithm
        to_draw.sort(key = lambda x: (x[0][0][2]+x[0][1][2]+x[0][2][2])/3, reverse=True)

        for tup in to_draw:
            proj_pts = tup[0]

            light_dot = tup[1]
            #Draw triangles
            xs = [proj_pts[0][0]+1, proj_pts[1][0]+1, proj_pts[2][0]+1]
            ys = [proj_pts[0][1]+1, proj_pts[1][1]+1, proj_pts[2][1]+1]
            #Scale
            xs = [i*width/2 for i in xs]
            ys = [j*height/2 for j in ys]
            pygame.gfxdraw.filled_trigon(screen, int(xs[0]), height-int(ys[0]), int(xs[1]), height-int(ys[1]), int(xs[2]), height-int(ys[2]), (colors[0][0]*light_dot, colors[0][1]*light_dot, colors[0][2]*light_dot))
            if drawMesh:
                pygame.draw.line(screen, (255, 255, 255), (xs[0], ys[0]), (xs[1], ys[1]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[1], ys[1]), (xs[2], ys[2]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[2], ys[2]), (xs[0], ys[0]), line_width)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
