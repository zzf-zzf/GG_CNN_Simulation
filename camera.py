import math
import numpy as np

# image size(displaying)
Height_img = 480
Width_img = 640

def Radians2Angle(radians):
    return 180 * radians / math.pi

def Angle2Radians(angle):
    return angle * math.pi / 180

def Euler2Roatation(theta):
    """
    transfer Euler angle to rotation matrix
    :param theta: [roll, pitch, yaw]
    :return:
    """
    R_X = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),  math.cos(theta[0])]
                  ])

    R_Y = np.array([[math.cos(theta[1]),  0, math.sin(theta[1])],
                    [0,                   1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_Z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),  math.cos(theta[2]), 0],
                    [                 0,                   0, 1]
                    ])
    Rotation_Matrix = np.dot(R_Z, np.dot(R_Y, R_X))
    return Rotation_Matrix

def GetT(translation, rotate):
    """
    get the transform matrix about rotate and translation
    :param translation: (x, y, z)
    :param rotate: rotate matrix
    :return:
    """
    T_matrix = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], translation[0]],
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], translation[1]],
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], translation[2]],
        [           0,            0,            0,              1.]
    ])
    return T_matrix


class Camera:
    def __init__(self):
        """
        Initialize the parameter of camera, calculate the inner parameter of camera
        """
        self.fov = 70 #field of view(vetical)， should be optimized
        self.height = 0.5 # height of camera
        self.RealHigh = self.height * math.tan(Angle2Radians(self.fov/2)) #the real distance between the mid-point of image and edge
        self.RealWidth = Width_img * self.RealHigh / Height_img #the real width of image

        # calculate focal length fx and fy (for square image, fx=fy)
        self.f = (Height_img / 2) * self.height / self.RealHigh

        # inner parameter
        self.InMatrix = np.array([[self.f, 0, Width_img/2-0.5], [0, self.f, Height_img/2-0.5], [0, 0, 1]], dtype=np.float)

        # calculate the transfer matrix from world coordinate to camera coordinate  4*4
        RotMatrix = Euler2Roatation([math.pi, 0, 0])
        self.T_Matrix = GetT([0, 0, self.height], RotMatrix)

    def Camera_Height(self):
        return self.height

    def Img2Camera(self, loc, depth):
        """
        Get the coordinates of the pixel point 'loc' in the camera coordinate system
        :param loc:(x, y)
        :param depth:
        :return:[x, y, z]
        """
        # print(loc)
        locInImage = np.array([[loc[0]], [loc[1]], [1]], dtype=np.float)

        Ret = np.matmul(np.linalg.inv(self.InMatrix), locInImage) * depth
        return list(Ret.reshape((3,)))

    def Camera2Image(self, coordinate):
        """
        Convert points in camera coordinate system to image
        :param coordinate: [x, y, z]
        :return: [row, col]
        """
        z_axis = coordinate[2]
        coordinate = np.array(coordinate).reshape(3, 1)
        loc_img = (np.matmul(self.InMatrix, coordinate) / z_axis).reshape((3,))
        return list(loc_img)[:-1]

    def Line2ImagePixel(self, l, depth):
        """
        calculate a line's pixels in image
        :param l:
        :param depth:
        :return:
        """
        return l * self.f / depth

    def ImagePixel2Line(self, pixel, depth):
        """

        :param pixel:
        :param depth:
        :return:
        """
        return pixel * depth / self.f

    def Camera2World(self, coordinate):
        """
        camera frame to world frame
        :param coordinate:
        :return:
        """
        coordinate.append(1.)
        coordinate = np.array(coordinate).reshape((4, 1))
        coordinate_world = np.matmul(self.T_Matrix, coordinate).reshape((4,))
        return list(coordinate_world)[:-1]

    def World2Camera(self, coordinate):
        """

        :param coordinate:
        :return:
        """
        coordinate.append(1.)
        coordinate = np.array(coordinate).reshape((4, 1))
        coordinate_world = np.matmul(np.linalg.inv(self.T_Matrix), coordinate).reshape((4,))
        return list(coordinate_world)[:-1]

    def World2Image(self, coordinate):
        """

        :param coordinate:
        :return:
        """
        coordinate = self.World2Camera(coordinate)
        loc = self.Camera2Image(coordinate) #[y, x}
        return [int(loc[1]), int(loc[0])]

    def Image2World(self, loc, depth):
        """

        :param loc:
        :param depth:
        :return:
        """
        # print(loc)
        CorInCamera = self.Img2Camera(loc, depth)
        return self.Camera2World(CorInCamera)



if __name__ == '__main__':
    camera = Camera()
    print(camera.InMatrix)