"""

"""

import pybullet as p
import pybullet_data
import math
import os
import random
import cv2
import numpy as np
import shutil #realize some function like file copying, moving, compressing and decompressing
import sys
sys.path.append('/home/zf/ggcnn_self_sim')

# image size(depth image), affect the grasp effect(like location of grasp), should be optimized
image_width = 640
image_height = 480 #the size should equal to training dataset
near_plane = 0.01
far_plane = 5
fov = 70 # height of camera
aspect = image_width / image_height

#def img_resize(image, size, interp="nearest"):


#From google
def image_inpaint(img, missing_value=0):
    """
    inpaint the missing value of depth image
    :param img: the depth image that should be filled
    :param missing_value: value to fill the depth image
    :return: depth image
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT) #set the bounding box of the image
    mask = (img == missing_value).astype(np.unit8)

    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # turn back to original size and value range
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


class Env_Sim(object):
    """
    the class of virtual environment
    """
    def __init__(self, client, path, gripperId=None):
        """
        :param client:
        :param path: the path of model
        :param gripperId:
        """
        self.zzf = client
        self.zzf.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, solverResidualThreshold=0, enableFileCaching=0)
        self.zzf.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0]) # chenge of the camera's vision
        self.zzf.setAdditionalSearchPath(pybullet_data.getDataPath()) # add the path
        self.planeId = self.zzf.loadURDF("plane.urdf", [0, 0, 0]) #load the ground
        self.trayId = self.zzf.loadURDF("/home/zf/bullet3-master/data/tray/tray.urdf", [0, -1.2, 0]) #load the tray
        self.zzf.setGravity(0, 0, -9.8) #add gravity acceleration
        self.flags = self.zzf.URDF_ENABLE_CACHED_GRAPHICS_SHAPES #
        self.gripperId = gripperId #variable, select the gripper engine when run the main function

        #load the camera
        self.movecamera(0, 0)
        self.projectionMatrix = self.zzf.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

        #get the 'list' file from the path
        list_file = os.path.join(path, 'list')
        if not os.path.exists(list_file):
            raise shutil.Error
        self.urdfs_list = []
        with open(list_file, 'r') as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                self.urdfs_list.append(os.path.join(path, line[:-1]+'.urdf'))

        self.urdfs_num = 0
        self.urdfs_id = []
        self.object_id = []
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]



    def movecamera(self, x, y, z=0.7):
        """
        move the camera to the specific location
        :param x: coordinate of x at world coordinate system
        :param y: coordinate of y at the world coordinate system
        :param z: coordinate of z at the world coordinate system (default=0.7)
        :return:
        """
        self.viewMatrix = self.zzf.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0]) #

    def num_urdfs(self):
        return len(self.urdfs_list)

    def loadURDFObject(self, idx, num):
        """
        load the objects should be grasped
        :param idx: index umber
        :param num: the number of loading objects
        :return:
        """
        assert idx >=0 and idx < len(self.urdfs_list)
        self.urdfs_num = num

        # get the object's file
        if (idx + self.urdfs_num - 1) > (len(self.urdfs_list) - 1):
            self.urdfs_name = self.urdfs_list[idx:]
            self.urdfs_name += self.urdfs_list[:len(self.urdfs_list)-idx]
            self.object_id = list(range(idx, len(self.urdfs_list)))
            self.object_id += list(range(self.urdfs_num + idx - len(self.urdfs_list)))
        else:
            self.urdfs_name = self.urdfs_list[idx:idx+self.urdfs_num]
            self.object_id = list(range(idx, idx+self.urdfs_num))

        print('self.object_id = \n', self.object_id)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []

        for i in range(self.urdfs_num):
            pos_obj = 0.1
            basePosition = [random.uniform(-1*pos_obj, pos_obj), random.uniform(-1*pos_obj, pos_obj), random.uniform(0.2, 1)]
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.zzf.getQuaternionFromEuler(baseEuler)

            urdf_id = self.zzf.loadURDF(self.urdfs_name[i], basePosition, baseOrientation)

            # define collision
            if self.gripperId is not None:
                self.zzf.setCollisionFilterPair(urdf_id, self.gripperId, -1, 0, 1)
                self.zzf.setCollisionFilterPair(urdf_id, self.gripperId, -1, 1, 1)
                self.zzf.setCollisionFilterPair(urdf_id, self.gripperId, -1, 2, 1)

            inf = self.zzf.getVisualShapeData(urdf_id)[0]
            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5])
            self.urdfs_scale.append(inf[3][0])

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break

    # scene 1 : load a desk that higher than z_thresh, then let the robotic arm grasp object and put it on the desk
    def Eval_Grasp1(self, z_thresh):
        """
        exam whether this grasp is successful
        :param z_thresh: the criterion of evaluation
        :return:
        """
        for i in range(self.urdfs_num):
            loc, ori = self.zzf.getBasePositionAndOrientation(self.urdfs_id[i])
            if loc[2] >= z_thresh:
                return True
            print('======================This grasp is failed======================')
            return False

    # scene 2:judge whether the grasping object is put on the tray(can be a range of location)
    #def Eval_Grasp2(self, ):


    # should change to better
    def Render_Depth_Image(self):
        """
        rendering the needed images about grasping
        :return:image_depth
        """
        #rendering
        img_camera = self.zzf.getCameraImage(image_width, image_height, self.viewMatrix, self.projectionMatrix,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w_pixel = img_camera[0]
        h_pixel = img_camera[1]
        deep_pixel = img_camera[3]

        #get the depth image
        depth = np.reshape(deep_pixel, (h_pixel, w_pixel))
        A = np.ones((image_height, image_width), dtype=np.float64) * far_plane * near_plane
        B = np.ones((image_height, image_width), dtype=np.float64) * far_plane
        C = np.ones((image_height, image_width), dtype=np.float64) * (far_plane - near_plane)
        image_depth = np.divide(A, (np.subtract(B, np.multiply(C, depth))))
        return image_depth

    def Render_Camera_Mask(self):
        """
        rendering the needed image that calculate data about grasping
        :return: img_mask
        """
        img_camera = self.zzf.getCameraImage(image_width, image_height, self.viewMatrix, self.projectionMatrix,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w_pixel = img_camera[0]
        h_pixel = img_camera[1]
        mask = img_camera[4]

        img_mask = np.reshape(mask, (h_pixel, w_pixel)).astype(np.unit8)
        img_mask[img_mask > 2] = 255
        return img_mask

























