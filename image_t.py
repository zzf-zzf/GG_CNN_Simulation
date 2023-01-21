"""

"""
import numpy as np

def Depth2Gray_3(image_depth):
    """

    :param image_depth:
    :return:
    """
    max_x = np.max(image_depth)
    min_x = np.min(image_depth)
    if min_x == max_x:
        print('!!!!Some error occur when rendering image!!!!')
        raise EOFError
    k = 255 / (max_x - min_x)
    b = 255 - k * max_x
    ret = (image_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret