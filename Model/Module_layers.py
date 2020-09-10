import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x

class Conv_Layer_box(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation_func, batch_normalization):
        super().__init__()
        padding = (int((kernel_size - 1)/2), int((kernel_size - 1)/2))
        dict_activation_func = {"ReLU": nn.ReLU(inplace=False),
                                "linear": nn.ReLU(inplace=False),
                                "leaky": nn.LeakyReLU(0.1, inplace=False),
                                "mish": Mish()
                               }
        
        if batch_normalization == True:
            bias = False
        else:
            bias = True
        self.conv_box = nn.ModuleList()
        self.conv_box.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias = bias))
        if batch_normalization == True:
            self.conv_box.append(nn.BatchNorm2d(out_channel))
        if activation_func != "linear":
            self.conv_box.append(dict_activation_func[activation_func])
        
    def forward(self, x):
        for layer in self.conv_box:
            x = layer(x)
        return x

class Maxpool_pad_Layer_box(nn.Module):
    def __init__(self, maxpool_size):
        super().__init__()
        self.maxpool_size = maxpool_size
        #why there are 2 padding??????????????
        self.pad_1 = int((self.maxpool_size - 1) / 2)
        self.pad_2 = self.pad_1
    def forward(self, x):
        x = F.pad(x, (self.pad_1, self.pad_2, self.pad_1, self.pad_2), mode = 'replicate')
        x = F.max_pool2d(x, self.maxpool_size, stride = 1)
        return x

class Upsample_layer(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        
    def forward(self, x):
        batch, channel, height, width = x.data.size()
        x = x.view(batch, channel, height, 1, width, 1).expand(batch, channel, height, self.stride, width, self.stride).clone()
        x = x.contiguous().view(batch, channel, height * self.stride, width * self.stride).clone()
        return x

class shortcut(nn.Module):
    def __init__(self):
        super().__init__()
        
class route(nn.Module):
    def __init__(self):
        super().__init__()
		

class Yolo_TL(nn.Module):
    def __init__(self, anchors, mask, classes, input_image_size):
        super().__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.number_of_mask = len(mask)
        self.input_image_size = input_image_size
        #self.Sigmoid_layer = nn.Sigmoid()

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
        
        #c_x = [i for i in range(x.size(2))]
        #c_y = [i for i in range(x.size(2))]
        #add_on_Matrix_x = torch.from_numpy(np.array([[i for j in range(x.size(2))] for i in range(x.size(2))])).cuda()
        #add_on_Matrix_y = torch.from_numpy(np.array([[i for i in range(x.size(2))] for j in range(x.size(2))])).cuda()
        
        b_x = [None for i in range(number_of_mask)]
        b_y = [None for i in range(number_of_mask)]
        b_w = [None for i in range(number_of_mask)]
        b_h = [None for i in range(number_of_mask)]
        
        anchor_shape_1 = int(len(anchors) / 2)
        anchors = np.array(anchors).reshape(anchor_shape_1, 2)
        
        for i in range(number_of_mask):
            start_point = i * (5 + classes)
            end_point = (i + 1) * (5 + classes)
            
            t_x[i] = x[:, (start_point + 0) : (start_point + 1), :, :].clone()
            t_y[i] = x[:, (start_point + 1) : (start_point + 2), :, :].clone()
            t_w[i] = x[:, (start_point + 2) : (start_point + 3), :, :].clone()
            t_h[i] = x[:, (start_point + 3) : (start_point + 4), :, :].clone()
            objective_p[i] = x[:, (start_point + 4) : (start_point + 5), :, :].clone()
            class_p[i] = x[:, (start_point + 5) : end_point, :, :].clone()

            b_x[i] = torch.sigmoid(t_x[i].clone())
            b_y[i] = torch.sigmoid(t_y[i].clone())
            b_w[i] = torch.exp(t_w[i].clone())
            b_h[i] = torch.exp(t_h[i].clone())
            
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
		
		
class Yolo_TL_prediction(nn.Module):
    def __init__(self, anchors, mask, classes, input_image_size):
        super().__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.number_of_mask = len(mask)
        self.input_image_size = input_image_size
        #self.Sigmoid_layer = nn.Sigmoid()

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

#build module for entire YOLO
class YOLO_v4_model(nn.Module):
    def __init__(self, layer_details, layer_type):
        super(YOLO_v4_model, self).__init__()
        self.all_layers = nn.ModuleList()
        all_layers = self.all_layers
        self.layer_details = layer_details
        self.layer_type = layer_type

        for i in range(len(layer_type)):
            if layer_type[i] == 'convolutional':
                try:
                    if int(layer_details[i]['batch_normalize']) == 1:
                        batch_normalize = True
                    else:
                        batch_normalize = False
                except:
                    batch_normalize = False
                if i == 0:
                    in_channel = 3
                else:
                    in_channel = None
                    if layer_type[i - 1] == 'convolutional':
                        skip_step = [0]
                    elif layer_type[i - 1] == 'shortcut':
                        skip_step = [int(layer_details[i - 1]['from'])]
                    elif layer_type[i - 1] == 'route':
                        skip_step = layer_details[i - 1]['layers'].split(",")

                    for SS in skip_step:
                        SS = int(SS)
                        if SS > 0:
                            if in_channel == None:
                                in_channel = int(layer_details[SS]['filters'])
                            else:
                                in_channel += int(layer_details[SS]['filters'])
                        else:
                            if in_channel == None:
                                in_channel = int(layer_details[i - 1 + SS]['filters'])
                            else:
                                in_channel += int(layer_details[i - 1 + SS]['filters'])
                out_channel = int(layer_details[i]['filters'])
                kernel_size = int(layer_details[i]['size'])
                stride = int(layer_details[i]['stride'])
                pad = int(layer_details[i]['pad'])
                activation_func = layer_details[i]['activation']
                layer = Conv_Layer_box(in_channel, out_channel, kernel_size, stride, activation_func, batch_normalize)
            elif layer_type[i] == 'maxpool':
                layer_details[i].update([('filters', layer_details[i - 1]['filters'])])
                maxpool_size = int(layer_details[i]['size'])
                layer = Maxpool_pad_Layer_box(maxpool_size)
            elif layer_type[i] == 'upsample':
                layer_details[i].update([('filters', layer_details[i - 1]['filters'])])
                stride = int(layer_details[i]['stride'])
                layer = Upsample_layer(stride)
            elif layer_type[i] == 'yolo':
                anchors = [int(x) for x in layer_details[i]['anchors'].split(",")]
                mask = [int(x) for x in layer_details[i]['mask'].split(",")]
                classes = int(layer_details[i]['classes'])
                layer = Yolo(anchors, mask, classes, input_image_size)

            elif layer_type[i] == 'shortcut':
                skip_step = int(layer_details[i]['from'])
                layer_details[i].update([('filters', layer_details[i + skip_step]['filters'])])
                layer = shortcut()
            elif layer_type[i] == 'route':
                try:
                    skip_step = int(layer_details[i]['layers'].split(",")[0])
                except:
                    skip_step = int(layer_details[i]['layers'])
                #print(skip_step)
                if skip_step > 0:
                    layer_details[i].update([('filters', layer_details[skip_step]['filters'])])
                else:
                    layer_details[i].update([('filters', layer_details[i + skip_step]['filters'])])
                layer = route()
            elif layer_type[i] == 'net':
                #print("net")
                continue
            else:
                continue
            all_layers.append(layer)
        global all_layerrr
        all_layerrr = all_layers

    def forward(self, x):
        all_layers = self.all_layers
        layers_output = [None for i in range(len(layer_type))]
        for i in range(len(layer_type)):
            #print(i)
            if i == 0:
                layers_output[i] = all_layers[i](x)
                continue
            elif layer_type[i] == 'yolo':
                layers_output[i] = all_layers[i](layers_output[i - 1])
                continue
            elif layer_type[i] == 'convolutional' or layer_type[i] == 'maxpool' or layer_type[i] == 'upsample' or layer_type[i] == 'yolo':
                layers_output[i] = all_layers[i](layers_output[i - 1])
                continue
            elif layer_type[i] == 'shortcut':
                skip_step = [int(layer_details[i]['from'])]
            elif layer_type[i] == 'route':
                skip_step = layer_details[i]['layers'].split(",")
            for SS in skip_step:
                SS = int(SS)
                if SS > 0:
                    if layers_output[i] == None:
                        layers_output[i] = layers_output[SS]
                    else:
                        layers_output[i] = torch.cat((layers_output[i], layers_output[SS]), 1)
                else:
                    if layers_output[i] == None:
                        layers_output[i] = layers_output[i + SS]
                    else:
                        
                        #print(layers_output[i + SS].size())
                        layers_output[i] = torch.cat((layers_output[i], layers_output[i + SS]), 1)
        return layers_output[137], layers_output[148], layers_output[159]



