import numpy as np
import itertools

class Cuboid:

    def __init__(self, origin, orientation, length):
        self.origin = origin
        self.orientation = orientation
        self.length = length

    def get_edges_normals(self,origin,orientation,length):
        theta_x,theta_y,theta_z = orientation

        sz, cz = np.sin(theta_z), np.cos(theta_z)
        sy, cy = np.sin(theta_y), np.cos(theta_y)
        sx, cx = np.sin(theta_x), np.cos(theta_x)

        Rz = np.array(((cz,-sz,0),(sz,cz,0),(0,0,1)))
        Ry = np.array(((cy,0,sy),(0,1,0),(-sy,0,cy)))
        Rx = np.array(((1,0,0),(0,cx,-sx),(0,sx,cx)))

        R = np.matmul(Rz,(np.matmul(Ry,Rx)))

        end_a = np.subtract(origin,np.true_divide(length,2))
        end_b = np.add(origin,np.true_divide(length,2))

        x_ends = [end_a[0],end_b[0]]
        y_ends = [end_a[1],end_b[1]]
        z_ends = [end_a[2],end_b[2]]

        vertices = []

        for i in itertools.product(x_ends,y_ends,z_ends):
            vertices.append(i)

        vertices = np.vstack(vertices).T
        vertices = np.matmul(R,vertices)
        #return vertices of the form 3xN where N is the number of edges and surface normals in
        # column form (3x1)

        return R, vertices

    def projection(self,axes,vertices1,vertices2):
        vertices1 = vertices1.T
        vertices2 = vertices2.T
        for i in range(axes.shape[0]):
            axis = axes[i].reshape(3,1)
            projection1 = np.dot(vertices1,axis)
            projection2 = np.dot(vertices2,axis)

            max_1 = np.amax(projection1)
            max_2 = np.amax(projection2)
            min_1 = np.amin(projection1)
            min_2 = np.amin(projection2)

            if(min_1>max_2 or max_1<min_2):
                return False

        return True

    def get_axes(self,normals1,normals2):
        normals1 = normals1.T                #reshape normals to 1x3
        normals2 = normals2.T
        axes = []
        axes.append(normals1)
        axes.append(normals2)
        axes.append(np.cross(normals1[0],normals2))
        axes.append(np.cross(normals1[1],normals2))
        axes.append(np.cross(normals1[2],normals2))

        axes = np.vstack(axes)   #axes are of the shape (15x3)


        return axes

    def check_collision(self,cuboid2):

        normals1, vertices1 = self.get_edges_normals(self.origin,self.orientation,self.length) #vertices are stacked as 3xN
        normals2, vertices2 = cuboid2.get_edges_normals(cuboid2.origin,cuboid2.orientation,cuboid2.length)
        axes = self.get_axes(normals1,normals2)
        collision = self.projection(axes,vertices1,vertices2)

        return collision


def main():


    cuboid1 = Cuboid(np.array((0,0,0)),np.array((0,0,0)),np.array((3,1,2)))

    origin = np.array(((0,1,0),(1.5,-1.5,0),(0,0,-1),(3,0,0),(-1,0,-2),(1.8,0.5,1.5),(0,-1.2,0.4),(-0.8,0,-0.5)))
    orientation = np.array(((0,0,0),(1,0,1.5),(0,0,0),(0,0,0),(0.5,0,0.4),(-0.2,0.5,0),(0,0.785,0.785),(0,0,0.2)))
    length = np.array(((0.8,0.8,0.8),(1,3,3),(2,3,1),(3,1,1),(2,0.7,2),(1,3,1),(1,1,1),(1,0.5,0.5)))

    for i in range(origin.shape[0]):
        cuboid2 = Cuboid(origin[i],orientation[i],length[i])
        collision = cuboid1.check_collision(cuboid2)
        print(collision)


if __name__ == '__main__':
    main()
