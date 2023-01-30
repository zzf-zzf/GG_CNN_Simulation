import cv2
import torch
import math
import numpy as np
from ggcnn_base.models.common import post_process_output
from ggcnn_base.models.ggcnn import GGCNN
from skimage.draw import line, polygon


def Draw_Rect(p1, p2, w):
    """

    :param p1:
    :param p2:
    :param w:
    :return:
    """
    y1, x1 = p1
    y2, x2 = p2

    if x1 == x2: # should be optimized
        if y1 > y2:
            Angle = math.pi / 2
        else:
            Angle = 3 * math.pi / 2
    else:
        tan_angle = (y1 - y2) / (x2 - x1)
        Angle = np.arctan(tan_angle)

    points = []
    points.append([y1 - w / 2 * np.cos(Angle), x1 - w / 2 * np.sin(Angle)])
    points.append([y2 - w / 2 * np.cos(Angle), x2 - w / 2 * np.sin(Angle)])
    points.append([y2 + w / 2 * np.cos(Angle), x2 + w / 2 * np.sin(Angle)])
    points.append([y1 + w / 2 * np.cos(Angle), x1 + w / 2 * np.sin(Angle)])
    points = np.array(points)

    ROWs, COLs = polygon(points[:, 0], points[:, 1], (10000, 10000)) # get all rows and cols of points at grasp rect
    return ROWs, COLs

def Opp_Angle(Angle):
    """

    :param Angle:
    :return:
    """
    return Angle + math.pi

def Draw_Grasp(image, grasps, mode='line'):
    """
    draw the grasp line
    :param image:
    :param grasps:
    :param mode:
    :return:
    """
    assert mode in ['line', 'region']

    Num = len(grasps)
    for i, grasp in enumerate(grasps):
        Row, Col, Angle, Width = grasp
        color_b = 255 / Num * i
        color_r = 0
        color_g = -255 / Num * i + 255
        if mode == 'line':
            Width = Width / 2
            OppAngle = Opp_Angle(Angle)
            k = math.tan(Angle)

            if k == 0:
                dx = Width
                dy = 0
            else:
                dx = k / abs(k) * Width / pow(k ** 2 + 1, 0.5)
                dy = k * dx # triangle function

            if Angle < math.pi:
                cv2.line(image, (Col, Row), (int(Col + dx), int(Row - dy)), (255, 0, 0), 1)
            else:
                cv2.line(image, (Col, Row), (int(Col - dx), int(Row + dy)), (255, 0, 0), 1)

            if OppAngle < math.pi:
                cv2.line(image, (Col, Row), (int(Col + dx), int(Row - dy)), (255, 0, 0), 1)
            else:
                cv2.line(image, (Col, Row), (int(Col - dx), int(Row + dy)), (255, 0, 0), 1)

            cv2.circle(image, (Col, Row), 1, (color_b, color_g, color_r), -1)
        else:
            image[Row, Col] = [color_b, color_g, color_r]

    return image

def Input_Image(image, out_size=300):
    """
    reshape the image, keep the center image
    :param image:
    :param out_size:
    :return:
    """
    assert image.shape[0] >= out_size and image.shape[1] >= out_size
    cut_x1 = int((image.shape[1] - out_size) / 2)
    cut_x2 = cut_x1 + out_size
    cut_y1 = int((image.shape[0] - out_size) / 2)
    cut_y2 = cut_y1 + out_size
    image = image[cut_y1:cut_y2, cut_x1:cut_x2]

    image = np.clip(image - image.mean(), 0., 1.).astype(np.float32)

    Tensor = torch.from_numpy(image[np.newaxis, np.newaxis, :, :]) # data type transform

    return Tensor, cut_x1, cut_y1

def Whether_Collision(point, dep, angle, depth_img, finger1, finger2):
    """

    :param point:
    :param dep:
    :param angle:
    :param depth_img:
    :param finger1:
    :param finger2:
    :return:
    """
    Row, Col = point

    Row1 = int(Row - finger2 * math.sin(angle))
    Col1 = int(Col + finger2 * math.cos(angle))

    ROWs, COLs = Draw_Rect([Row, Col], [Row1, Col1], finger1)

    if np.min(depth_img[ROWs, COLs]) > dep: #without collision
        return True
    return False

def Grasp_Depth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger1, finger2):
    """

    :param camera_depth:
    :param grasp_row:
    :param grasp_col:
    :param grasp_angle:
    :param grasp_width:
    :param finger1:
    :param finger2:
    :return:
    """
    k = math.tan(grasp_angle)
    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx

    point1 = (int(grasp_row - dy), int(grasp_col + dx))
    point2 = (int(grasp_row + dy), int(grasp_col - dx))

    rr, cc = line(point1[0], point1[1], point2[0], point2[1])
    min_depth = np.min(camera_depth[rr, cc])

    grasp_depth = min_depth + 0.005
    while grasp_depth < min_depth + 0.05:
        if not Whether_Collision(point1, grasp_depth, grasp_angle, camera_depth, finger1, finger2):
            return grasp_depth - 0.005
        if not Whether_Collision(point2, grasp_depth, grasp_angle + math.pi, camera_depth, finger1, finger2):
            return grasp_depth - 0.005
        grasp_depth += 0.005

    return grasp_depth

def Get_Predict(net, xc):
    """

    :param net:
    :param xc:
    :return:
    """
    net.eval()
    with torch.no_grad():
        pred_pos, pred_cos, pred_sin, pred_width = net(xc)

        pred_pos = torch.sigmoid(pred_pos)
        pred_cos = torch.sigmoid(pred_cos)
        pred_sin = torch.sigmoid(pred_sin)
        pred_width = torch.sigmoid(pred_width)

    return pred_pos, pred_cos, pred_sin, pred_width


class GGCNNNet:
    def __init__(self, model, device):
        """

        :param model:
        :param device:
        """
        self.device = device
        self.net = GGCNN()
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)
        self.net = self.net.to(device)
        print('load done')

    def Predict(self, image, input_size=320):
        """

        :param image:
        :param input_size:
        :return:
        """
        Input, self.cut_x1, self.cut_y1 = Input_Image(image, input_size)
        self.outpos, self.outcos,self.outsin, self.outwidth = Get_Predict(self.net, Input.to(self.device))
        pos_pred, ang_pred, width_pred = post_process_output(self.outpos, self.outcos, self.outsin, self.outwidth)

        # select the best point
        Loc = np.argmax(pos_pred)
        # print(Loc)
        Row = Loc // pos_pred.shape[0]
        Col = Loc % pos_pred.shape[0]
        Angle = (ang_pred[Row, Col] + 2 * math.pi) % math.pi
        Width = width_pred[Row, Col]
        Row += self.cut_y1
        Col += self.cut_x1
        # print(Row)

        return Row, Col, Angle, Width


















