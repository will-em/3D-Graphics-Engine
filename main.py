import pygame
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
def main():
    pygame.init()
    height = 800
    width = 800
    screen = pygame.display.set_mode((width, height))
    done = False

    clock = pygame.time.Clock()

    line_width = 2

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

    #Projection Matrix
    fNear = 0.1
    fFar = 1000.0
    fFov = 90.0
    fAspectRation = height/width
    fFovRad = 1.0/np.tan(np.deg2rad(fFov/2))

    projMat = np.array([[fAspectRation*fFovRad, 0, 0, 0], [0, fFovRad, 0, 0], [0, 0, fFar/(fFar-fNear), 1], [0, 0, (-fFar*fNear)/(fFar-fNear), 0]])



    theta=0
    while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True


        screen.fill((0, 0, 0))
        #Draw triangles

        theta+=0.01
        rotxMat = np.array([[1, 0, 0, 0], [0, np.cos(theta/2), np.sin(theta/2), 0], [0,-np.sin(theta/2), np.cos(theta/2), 0], [0, 0, 0, 1]])
        rotzMat = np.array([[np.cos(theta), np.sin(theta), 0, 0], [-np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for tri in meshCube.tris:

            #Translate & Rotate

            triRotatedX = copy.deepcopy(tri)
            triRotatedX.p[0]=MatrixMulti(rotxMat, triRotatedX.p[0])
            triRotatedX.p[1]=MatrixMulti(rotxMat, triRotatedX.p[1])
            triRotatedX.p[2]=MatrixMulti(rotxMat, triRotatedX.p[2])
            triRotatedZ = copy.deepcopy(triRotatedX)
            triRotatedZ.p[0]=MatrixMulti(rotzMat, triRotatedZ.p[0])
            triRotatedZ.p[1]=MatrixMulti(rotzMat, triRotatedZ.p[1])
            triRotatedZ.p[2]=MatrixMulti(rotzMat, triRotatedZ.p[2])

            triTrans = copy.deepcopy(triRotatedZ)
            dz = 3
            triTrans.p[0][:,2]=triTrans.p[0][:,2]+dz
            triTrans.p[1][:,2]=triTrans.p[1][:,2]+dz
            triTrans.p[2][:,2]=triTrans.p[2][:,2]+dz


            projPts = []
            for point in triTrans.p:
                projPts.append(MatrixMulti(projMat, point))

            #Draw triangles
            pt1 = projPts[0]
            pt2 = projPts[1]
            pt3 = projPts[2]
            x1 = (pt1[:,0]+1)*width/2
            y1 = (pt1[:,1]+1)*height/2
            x2 = (pt2[:,0]+1)*width/2
            y2 = (pt2[:,1]+1)*height/2
            x3 = (pt3[:,0]+1)*width/2
            y3 = (pt3[:,1]+1)*height/2

            pygame.draw.line(screen, (0, 128, 255), (x1, y1), (x2, y2), line_width)
            pygame.draw.line(screen, (0, 128, 255), (x2, y2), (x3, y3), line_width)
            pygame.draw.line(screen, (0, 128, 255), (x3, y3), (x1, y1), line_width)


        pygame.display.flip()
        clock.tick(144)

main()
