import pygame
from pygame import gfxdraw
import numpy as np
import copy

class Triangle:
    def __init__(self, p):
        self.p = p #x, y, z

    def __str__(self):
        return str(self.points[0])

class Mesh:
    def __init__(self):
        pass

def MatrixMulti(mat, point):
    point = np.insert(point, 3, 1)
    point = np.expand_dims(point, axis=0)
    newP = point.dot(mat)
    w = newP[:,3]
    newP = newP[:, 0:3]
    if w!=0:
        newP = newP/w

    return newP

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


def main():
    pygame.init()
    height = 800
    width = 800
    screen = pygame.display.set_mode((width, height))
    done = False

    clock = pygame.time.Clock()

    line_width = 2
    """
    meshCube = Mesh()
    #South
    tri1 = Triangle([np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])])
    tri2 = Triangle([np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])])
    #East
    tri3 = Triangle([np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0])])
    tri4 = Triangle([np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0])])
    #North
    tri5 = Triangle([np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0])])
    tri6 = Triangle([np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])])
    #West
    tri7 = Triangle([np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0])])
    tri8 = Triangle([np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])])
    #Top
    tri9 = Triangle([np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])])
    tri10 = Triangle([np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])])
    #Bottom
    tri11 = Triangle([np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])])
    tri12 = Triangle([np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])])

    meshCube.tris =  [tri1, tri2, tri3, tri4, tri5, tri6, tri7, tri8, tri9, tri10, tri11, tri12]
    """
    meshCube = Mesh()
    meshCube.tris = loadMesh(r"C:\Users\willi\Documents\GitHub\3D-Graphics-Engine\VideoShip.txt")

    camera = [0, 0, 0] #Camera position in space

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

    theta=0
    dz = 8
    while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True


        screen.fill((0, 0, 0))

        theta+=0.01
        rotxMat = np.array([[1, 0, 0, 0], [0, np.cos(theta/2), np.sin(theta/2), 0], [0,-np.sin(theta/2), np.cos(theta/2), 0], [0, 0, 0, 1]])
        rotzMat = np.array([[np.cos(theta), np.sin(theta), 0, 0], [-np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        for k, tri in enumerate(meshCube.tris):

            triTrans = copy.deepcopy(tri)
            transPts = []
            projPts = []
            for point in triTrans.p:
                #Rotate
                point=MatrixMulti(rotxMat, point)
                point=MatrixMulti(rotzMat, point)
                #Translate
                point[:,2]=point[:,2]+dz

                transPts.append(point)
                projPts.append(MatrixMulti(projMat, point))

            vec1 = [transPts[1][:,0]-transPts[0][:,0], transPts[1][:,1]-transPts[0][:,1], transPts[1][:,2]-transPts[0][:,2]]
            vec2 = [transPts[2][:,0]-transPts[0][:,0], transPts[2][:,1]-transPts[0][:,1], transPts[2][:,2]-transPts[0][:,2]]

            normal = np.cross(vec1, vec2, axis=0)
            if np.linalg.norm(normal)!=0:
                normal = normal/np.linalg.norm(normal)

            diffVec = [triP-cameraP for triP, cameraP in zip(transPts[0], camera)]

            light_direction = [0, 0, -1] #Single Direction Light
            light_direction = light_direction/np.linalg.norm(light_direction)
            light_dot = np.dot(light_direction, normal)

            if np.dot(diffVec, normal)<0:
                #Draw triangles
                xs = [projPts[0][:,0]+1, projPts[1][:,0]+1, projPts[2][:,0]+1]
                ys = [projPts[0][:,1]+1, projPts[1][:,1]+1, projPts[2][:,1]+1]
                #Scale
                xs = [i*width/2 for i in xs]
                ys = [j*height/2 for j in ys]
                pygame.gfxdraw.filled_trigon(screen, xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], colors[0])
                pygame.draw.line(screen, (255, 255, 255), (xs[0], ys[0]), (xs[1], ys[1]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[1], ys[1]), (xs[2], ys[2]), line_width)
                pygame.draw.line(screen, (255, 255, 255), (xs[2], ys[2]), (xs[0], ys[0]), line_width)


        pygame.display.flip()
        clock.tick(60)

main()
