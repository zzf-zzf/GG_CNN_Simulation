"""

"""

import pybullet as p
import cv2
import sys
sys.path.append('/home/zf/ggcnn_self_sim') #add the location of self-module, this is a temporary path, the folloeing modules are come from this path
from env_sim import Env_Sim
from camera import Camera
import panda_sim as PandaSim
from ggcnn_sim_help import GGCNNNet, Draw_Grasp, Grasp_Depth
import image_t as T

FINGER1 = 0.15
FINGER2 = 0.15

def run(Database_Path, Start_Index, Object_Num):
    """

    :param Database_Path:
    :param Start_Index:
    :param Object_Num:
    :return:
    """
    p.connect(p.GUI) # initialize the simulation environment
    Panda = PandaSim.PandaSimAuto(p, [0, -0.6, 0]) # load the gripper arm
    Env = Env_Sim(p, Database_Path, Panda.PandaId) # load environment
    camera = Camera()
    GG_CNN = GGCNNNet('/home/zf/ggcnn_self_sim/ggcnn_base/output/models/221204_2000_training_example/'
                      'epoch_47_iou_0.75_statedict.pt', device="cuda")

    successful_grasp = 0
    all_grasp = 0
    continue_fail_grasp = 0

    # load objects
    Env.loadURDFObject(Start_Index, Object_Num)

    # give time to ready
    while True:
        for _ in range(240 * 5):
            p.stepSimulation()

    # rendering depth image
        Camera_Depth = Env.Render_Depth_Image()

    # predict the grasp
        row, col, angle, width_pixels = GG_CNN.Predict(Camera_Depth, input_size=320)
        real_width = camera.ImagePixel2Line(width_pixels, Camera_Depth[row, col])
    # print(row)
        grasp_x, grasp_y, grasp_z = camera.Image2World([col, row], Camera_Depth[row, col])
        finger1_pixel = camera.Line2ImagePixel(FINGER1, Camera_Depth[row, col])
        finger2_pixel = camera.Line2ImagePixel(FINGER2, Camera_Depth[row, col])
        grasp_depth = Grasp_Depth(Camera_Depth, row, col, angle, width_pixels, finger1_pixel, finger2_pixel)

        grasp_z = max(0.7 - grasp_depth, 0.)

        print('+' * 100)
        print('grasp information:')
        print('grasp_x = ', grasp_x)
        print('grasp_y = ', grasp_y)
        print('grasp_z = ', grasp_z)
        print('grasp_depth = ', grasp_depth)
        print('grasp_angle = ', angle)
        print('grasp_width = ', real_width)
        print('+' * 100)

    # configuration of grasp (plotting)
        Image_RGB = T.Depth2Gray_3((Camera_Depth))
        Image_Grasp = Draw_Grasp(Image_RGB, [[row, col, angle, width_pixels]], mode='line')
        cv2.imshow('Grasping-Image', Image_Grasp)
        cv2.waitKey(300) # control the showing duration time of imshow

    # grasping
        while True:
            p.stepSimulation()
            if Panda.process([grasp_x, grasp_y, grasp_z], angle, real_width / 2):
                break

        all_grasp += 1

        if Env.Eval_Grasp1(z_thresh=0.2):
            successful_grasp += 1
            continue_fail_grasp = 0
            if Env.urdfs_num == 0:
                p.disconnect()
                return successful_grasp, all_grasp
        else:
            continue_fail_grasp += 1
            if continue_fail_grasp == 5:
                p.disconnect()
                return successful_grasp, all_grasp

        Panda.Set_ArmPose([0.5, -0.6, 0.2])

if __name__ == "__main__":
    Start_Index = 8
    Object_Num = 5
    Database_Path = '/home/zf/ggcnn_self_sim/objs'
    successful_grasp, all_grasp = run(Database_Path, Start_Index, Object_Num)
    print('\n==========================Successful rate of grasping: {}/{}={}'.format(successful_grasp, all_grasp,
                                                                                     successful_grasp / all_grasp))












