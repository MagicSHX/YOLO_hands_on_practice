import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class Yolo(nn.Module):
    def __init__(self, anchors, mask, classes, input_image_size):
        super().__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.number_of_mask = len(mask)
        self.input_image_size = input_image_size

    def forward(self, x):
        mask = self.mask
        anchors = self.anchors
        classes = self.classes
        number_of_mask = self.number_of_mask
        input_image_size = self.input_image_size
        grid_size = int(input_image_size / x.size(2))

        t_x = [None for i in range(number_of_mask)]
        t_y = [None for i in range(number_of_mask)]
        t_w = [None for i in range(number_of_mask)]
        t_h = [None for i in range(number_of_mask)]
        objective_p = [None for i in range(number_of_mask)]
        class_p = [None for i in range(number_of_mask)]
        
        add_on_Matrix_x = torch.from_numpy(np.array([[i for j in range(x.size(2))] for i in range(x.size(2))])).cuda()
        add_on_Matrix_y = torch.from_numpy(np.array([[i for i in range(x.size(2))] for j in range(x.size(2))])).cuda()
        
        b_x = [None for i in range(number_of_mask)]
        b_y = [None for i in range(number_of_mask)]
        b_w = [None for i in range(number_of_mask)]
        b_h = [None for i in range(number_of_mask)]
        
        anchor_shape_1 = int(len(anchors) / 2)
        anchors = np.array(anchors).reshape(anchor_shape_1, 2)
        
        p_w = [None for i in range(number_of_mask)]
        p_h = [None for i in range(number_of_mask)]
        
        for i in range(number_of_mask):
            start_point = i * (5 + classes)
            end_point = (i + 1) * (5 + classes)
            p_w[i], p_h[i] = anchors[mask[i]]

            t_x[i] = x[:, (start_point + 0) : (start_point + 1), :, :].clone()
            t_y[i] = x[:, (start_point + 1) : (start_point + 2), :, :].clone()
            t_w[i] = x[:, (start_point + 2) : (start_point + 3), :, :].clone()
            t_h[i] = x[:, (start_point + 3) : (start_point + 4), :, :].clone()
            objective_p[i] = x[:, (start_point + 4) : (start_point + 5), :, :].clone()
            class_p[i] = x[:, (start_point + 5) : end_point, :, :].clone()

            b_x[i] = torch.sigmoid(t_x[i].clone()) + add_on_Matrix_x
            b_y[i] = torch.sigmoid(t_y[i].clone()) + add_on_Matrix_y

            #need to think whether need to use below 2 lines
            b_x[i] = grid_size * b_x[i].clone()
            b_y[i] = grid_size * b_y[i].clone()
            b_w[i] = p_w[i] * torch.exp(t_w[i].clone())
            b_h[i] = p_h[i] * torch.exp(t_h[i].clone())
            
            objective_p[i] = torch.sigmoid(objective_p[i].clone())
            class_p[i] = torch.sigmoid(class_p[i].clone())
        
        b_x = torch.stack(b_x).clone()
        b_y = torch.stack(b_y).clone()
        b_w = torch.stack(b_w).clone()
        b_h = torch.stack(b_h).clone()
        objective_p = torch.stack(objective_p).clone()
        class_p = torch.stack(class_p).clone()
        combined_yolo_output = torch.cat((b_x, b_y, b_w, b_h, objective_p, class_p), 2)
        return combined_yolo_output