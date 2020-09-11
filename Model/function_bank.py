import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
import glob

def read_cfg_file(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')

    layer_type = []
    layer_details = []
    current_layer_details = {}
    for line in lines:
        #print(line)
        if line == '':
            continue
        elif line[0] == '#':
            continue
        else:
            if (line[0] == '['):
                layer_type.append(line[1 : -1])
                if current_layer_details != {}:
                    layer_details.append(current_layer_details)
                    current_layer_details = {}
            else:
                current_layer_details.update([(line.split("=")[0].rstrip(), line.split("=")[1].lstrip())])
    layer_details.append(current_layer_details)
    return layer_type, layer_details

def cross_length(a_1, a_2, b_1, b_2):
    if a_1 <= b_1 and a_2 >= b_1:
        return (min(a_2, b_2) - b_1)
    elif a_1 <= b_1 and a_2 <= b_1:
        return 0
    else:
        return cross_length(b_1, b_2, a_1, a_2)

def IoU(x_GT, y_GT, w_GT, h_GT, x_PD, y_PD, w_PD, h_PD):
    area_of_I = cross_length(x_GT, x_GT + w_GT, x_PD, x_PD + w_PD) * cross_length(y_GT, y_GT + h_GT, y_PD, y_PD + h_PD)
    area_of_U = h_GT * w_GT + h_PD * w_PD - area_of_I
    return area_of_I / area_of_U

def axis_conversion(x_centre, y_centre, h, w):
    return (x_centre - h / 2, y_centre - w / 2, h, w)
	
	
	
def image_reader(image_path_list):
    from PIL import Image
    import numpy as np
    final_output_array = []
    for image_path in image_path_list:
        image = Image.open(image_path)
        image = image.resize((608, 608))
        image = np.array(image)
        image_array = np.array([0 for i in range(3 * 608 * 608)]).reshape(3, 608, 608)
        for i in range(3):
            image_array[i] = image[:, :, i]
        #print(image_array.shape)
        final_output_array.append(image_array)
    return np.array(final_output_array)