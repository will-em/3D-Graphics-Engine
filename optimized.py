import pygame
from pygame import gfxdraw
import numpy as np
import copy
import time
import operator
import math
import os
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

def MatrixMulti(A, point):
    x = [point[0], point[1], point[2], 1.0]
    #Matrix Mutliplication
    newP = [x[0]*A[0,0]+ x[1]*A[1,0]+ x[2]*A[2, 0]+ x[3]*A[3, 0],
    x[0]*A[0,1]+ x[1]*A[1,1]+ x[2]*A[2, 1]+ x[3]*A[3, 1],
    x[0]*A[0,2]+ x[1]*A[1,2]+ x[2]*A[2, 2]+ x[3]*A[3, 2],
    x[0]*A[0,3]+ x[1]*A[1,3]+ x[2]*A[2, 3]+ x[3]*A[3, 3]]
    w = newP[3]
    newP = newP[0:3]
    if w!=0:
        newP[:] = [p/w for p in newP]
    return newP

def vectorAdd(v1, v2):
    return [v_i1+v_i2 for v_i1, v_i2 in zip(v1,v2)]

def vectorSub(v1, v2):
    return [v_i1-v_i2 for v_i1, v_i2 in zip(v1,v2)]

def vectorScalMul(vector, scalar):
    return [scalar*element for element in vector]

def normalize(vec):
    return vec/np.linalg.norm(vec)

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
                tris.append(Triangle([vert1, vert2, vert3]))

    return tris

def matrixPointAt(pos, target, up):
    #Calculate new Forward direction
    newForward = [t-p for t, p in zip(target, pos)]
    newForward = newForward/(np.linalg.norm(newForward))#Addera 1e-16 ifall error

    #Calculate new Up direction
    a = np.dot(up, newForward)*newForward
    newUp = [uEl-aEl for uEl, aEl in zip(up, a)]
    newUp = newUp/(np.linalg.norm(newUp))#Addera 1e-16 ifall error

    #Calculate new Right Direction
    newRight = np.cross(newUp, newForward)

    mat = np.array([[newRight[0], newRight[1], newRight[2], 0.0],
    [newUp[0], newUp[1], newUp[2], 0],
    [newForward[0], newForward[1], newForward[2], 0.0],
    [pos[0], pos[1], pos[2], 1.0]])
    return mat

def matInv(m):
    matInv = np.array([[m[0][0], m[1][0], m[2][0], 0.0],
    [m[0][1], m[1][1], m[2][1], 0.0],
    [m[0][2], m[1][2], m[2][2], 0.0],
    [-(m[3][0]*m[0][0]+m[3][1]*m[0][1]+m[3][2]*m[0][2]),
    -(m[3][0]*m[1][0]+m[3][1]*m[1][1]+m[3][2]*m[1][2]),
    -(m[3][0]*m[2][0]+m[3][1]*m[2][1]+m[3][2]*m[2 ][2]), 1.0]])
    return matInv

def rotMatX(theta):
    return np.array([[1, 0, 0, 0], [0, np.cos(theta/2), np.sin(theta/2), 0], [0,-np.sin(theta/2), np.cos(theta/2), 0], [0, 0, 0, 1]])

def rotMatY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0], [0, 1, np.sin(theta/2), 0], [-np.sin(theta),0, np.cos(theta), 0], [0, 0, 0, 1]])

def rotMatZ(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0, 0], [-np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def intersectPlane(plane_p, plane_n, lineStart, lineEnd):
    plane_n = VectorNormalize(plane_n)
    plane_d = np.dot(plane_n, plane_p)
    ad = np.dot(lineStart, plane_n)
    bd = np.dot(lineEnd, plane_n)
    t = (-plane_d-ad)/(bd-ad)

    lineStartToEnd = vectorSub(lineEnd, lineStart)
    lineToIntersect = vectorScalMul(lineStartToEnd, t)

    return vectorAdd(lineStart, lineToIntersect)

def Triangle_ClipAgainstPlane(plane_p, plane_n, in_tri, out_tri1, out_tri2):
    plane_n = VectorNormalize(plane_n)



def main():
    pygame.init()
    height = 800
    width = 800
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
    name = input("Please choose the obj file you want to render: ")
    filename = os.path.join(dirname, name)
    meshCube.tris = loadMesh(filename)

    camera = np.array([0, 0, -1]) #Initial camera-position in space

    #Projection Matrix
    fNear = 0.1
    fFar = 1000.0
    fFov = 90.0
    fAspectRation = height/width
    fFovRad = 1.0/np.tan(np.deg2rad(fFov/2))

    projMat = np.array([[fAspectRation*fFovRad, 0, 0, 0], [0, fFovRad, 0, 0], [0, 0, fFar/(fFar-fNear), 1], [0, 0, (-fFar*fNear)/(fFar-fNear), 0]])

    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    color3 = (0, 0, 255)
    color4 = (255, 0, 255)
    color5 = (255, 255, 51)
    color6 = (102, 255, 255)
    colors = [color1, color1, color2, color2, color3, color3, color4, color4, color5, color5, color6, color6]

    theta = 0
    transDist = 10
    rotSpeed = 0.1
    dx = 0
    dy = 0
    dz = 0
    speed = 0.2

    velVec = np.array([0, 0, 0])
    lookDir = np.array([0, 0, 1])
    lookDir = normalize(lookDir)
    sensitivity = 0.005
    dMousex = 0
    dMousey = 0
    thetaX = 0
    thetaY = 0

    while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN :
                    if event.key == pygame.K_LEFT:
                        dx = -speed
                    elif event.key == pygame.K_RIGHT:
                        dx = speed

                    if event.key == pygame.K_UP:
                        dy = speed
                    elif event.key == pygame.K_DOWN:
                        dy = -speed
                    if event.key == pygame.K_w:
                        velVec = vectorScalMul(lookDir, speed)
                    elif event.key == pygame.K_s:
                        velVec = vectorScalMul(lookDir, -speed)

                    if event.key == pygame.K_ESCAPE:
                        done = True


                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        dx = 0
                    elif event.key == pygame.K_RIGHT:
                        dx = 0

                    if event.key == pygame.K_UP:
                        dy = 0
                    elif event.key == pygame.K_DOWN:
                        dy = 0
                    if event.key == pygame.K_w:
                        velVec = [0, 0, 0]
                    elif event.key == pygame.K_s:
                        velVec = [0, 0, 0]


        dMouse = pygame.mouse.get_rel()
        dMousex = dMouse[0]
        dMousey = dMouse[1]

        thetaY -= dMousex*sensitivity
        thetaX += dMousey*sensitivity

        camera[0] += dx
        camera[1] += dy
        camera[2] += dz
        camera = camera + velVec

        screen.fill((0, 0, 0))

        theta += rotSpeed
        rotzMat = rotMatZ(theta/2)
        rotxMat = rotMatX(theta)

        totMat = rotzMat.dot(rotxMat)

        upVec = np.array([0, 1, 0])
        target = np.array([0, 0, 1])

        cameraRot = rotMatY(thetaY)
        cameraRot = cameraRot.dot(rotMatX(thetaX))

        lookDir = MatrixMulti(cameraRot, target)

        target = vectorAdd(camera, lookDir)

        cameraMat = matrixPointAt(camera, target, upVec)

        viewMat = matInv(cameraMat)


        #t = time.time()
        toDraw = []
        for k, tri in enumerate(meshCube.tris):
            transPts = []
            viewedPts = []
            projPts = []
            for point in tri.p:
                #Rotate
                #print(totMat)
                point=MatrixMulti(totMat, point)
                #Translate
                point[2]=point[2]+transDist
                transPts.append(point)
                #Worldspace to Viewspace
                point = MatrixMulti(viewMat, point)
                viewedPts.append(point)
                #Project
                projPts.append(MatrixMulti(projMat, point))

            vec1 = [transPts[1][0]-transPts[0][0], transPts[1][1]-transPts[0][1], transPts[1][2]-transPts[0][2]]
            vec2 = [transPts[2][0]-transPts[0][0], transPts[2][1]-transPts[0][1], transPts[2][2]-transPts[0][2]]

            normal = np.cross(vec1, vec2, axis=0)
            normal = normal/(np.linalg.norm(normal)+1e-16)

            diffVec = vectorSub(transPts[0], camera)

            light_direction = [0, 0, -1] #Single Direction Light
            if np.dot(diffVec, normal)<0:
                light_direction = normalize(light_direction)
                light_dot = np.dot(light_direction, normal)
                if light_dot<0:
                    light_dot=0

                toDraw.append((projPts, light_dot))

        #Painter's algorithm
        toDraw.sort(key = lambda x: (x[0][0][2]+x[0][1][2]+x[0][2][2])/3, reverse=True)

        for tup in toDraw:
            projPts = tup[0]

            light_dot = tup[1]
            #Draw triangles
            xs = [projPts[0][0]+1, projPts[1][0]+1, projPts[2][0]+1]
            ys = [projPts[0][1]+1, projPts[1][1]+1, projPts[2][1]+1]
            #Scale
            xs = [i*width/2 for i in xs]
            ys = [j*height/2 for j in ys]
            pygame.gfxdraw.filled_trigon(screen, int(xs[0]), height-int(ys[0]), int(xs[1]), height-int(ys[1]), int(xs[2]), height-int(ys[2]), (colors[0][0]*light_dot, colors[0][1]*light_dot, colors[0][2]*light_dot))
            if drawMesh:
                pygame.draw.line(screen, (255, 255, 255), (xs[0], ys[0]), (xs[1], ys[1]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[1], ys[1]), (xs[2], ys[2]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[2], ys[2]), (xs[0], ys[0]), line_width)
        #print((time.time()-t))
        pygame.display.flip()
        clock.tick(60)

main()
© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

