import os
from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
import glob
import glob
from PIL import ImageTk, Image
import numpy as np
import xml.etree.ElementTree as ET
from numpy import savetxt
import gc

from Model.function_bank import *
from Model.Module_layers import *
from Model.Module_layers_backup import *

torch.autograd.set_detect_anomaly(True)
cfgfile = "C:/Users/HX/Desktop/yolov4.cfg"
model_file_path = "Model/model.pt"
TL_model_file_path = "Model/TL_model.pt"

anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
mask = [0, 1, 2]
classes = 80
input_image_size = 608


#read TOLOv4 model architecture from cfg file
layer_type, layer_details = read_cfg_file(cfgfile)
net_layer = layer_details[0]
layer_type = layer_type[1:]
layer_details = layer_details[1:]

print(len(layer_type))
print(len(layer_details))
print(layer_details[1])




# This is used to test end to end structure of YOLOv4
#need to adjust target size according to YOLOv4 output size before testin

input = np.array([1 for i in range(608 * 608 * 3 * 1)]).reshape(1, 3, 608, 608)
#target = np.array([0 for i in range(7 * 7 * 30)])

target = np.array([0 for i in range(3 * 19 * 19 * 85)])

input_tensor = torch.Tensor(input)
output_tensor = torch.Tensor(target)
input = input_tensor.cuda()
target = output_tensor.cuda()

#x = np.array([1 for i in range(608 * 608 * 3)]).reshape(1, 3, 608, 608)
#x = torch.tensor(x)

learning_rate = 0.08
epoch_size = 2
steps_for_printing_out_loss = 1

YOLO_v4_Module_WIP = YOLO_v4_model(layer_details, layer_type)

loss_function = nn.MSELoss()
optimizer = optim.SGD(YOLO_v4_Module_WIP.parameters(), lr = learning_rate)
YOLO_v4_Module_WIP.load_state_dict(torch.load("C:/Users/HX/Desktop/model_YOLOv4.pt")['state_dict'])
YOLO_v4_Module_WIP.eval()
YOLO_v4_Module_WIP.cuda()
"""
for name, param in YOLO_v4_Module_WIP.named_parameters():
    print('name: ', name)
    print(type(param))
    print('param.shape: ', param.shape)
    print('param.requires_grad: ', param.requires_grad)
    print('=====')
#transfer learning:


for name, param in model.named_parameters():
    if name in ['fc.weight', 'fc.bias']:
        param.requires_grad = True
    else:
        param.requires_grad = False
"""

def training_model():
    for i in range(1, epoch_size + 1):
        optimizer.zero_grad()
        output, b, c = YOLO_v4_Module_WIP(input.cuda())
        #print(output.size())
        #b_x, b_y, b_w, b_h, objective_p, class_p = YOLO_v4_Module_WIP(input.cuda())
        #output = b_x
        #loss = loss_function(output, target.reshape(output.size(0), output.size(1), output.size(2), output.size(3)))
        global output_tensor
        output_tensor = output
        loss = loss_function(output, target.reshape(output.size(0), output.size(1), output.size(2), output.size(3)))
        loss.backward()
        optimizer.step()
        if i % (steps_for_printing_out_loss) == 0:
            print('Loss (epoch: ' + str(i) + '): ' + str(loss.cpu().detach().numpy()))
    #torch.save({'state_dict': YOLO_v4_Module_WIP.state_dict(),'optimizer': optimizer.state_dict()}, model_file_path)


#training_model()
#torch.save({'output': output_tensor}, 'Model/output.pt')
#YOLO_v4_Module_WIP.state_dict()


#to use YOLOv4 layer to convert image into layer output from 137, 148, 159

#training_data_image_folder = "F:/FlyAI/UnderwaterDetection_roundA/train-A/image/"
training_data_image_folder = "F:/FlyAI/UnderwaterDetection_roundB/train-B/"
batch_size = 1

image_path_list = glob.glob(training_data_image_folder + "*.jpg")

batch_no = math.ceil(len(image_path_list) / batch_size)
print(batch_no)



#########convert original image into output from YOLOv4
"""
YOLO_v4_Module_WIP = YOLO_v4_model(layer_details, layer_type)
YOLO_v4_Module_WIP.cuda()
YOLO_v4_Module_WIP.load_state_dict(torch.load("C:/Users/HX/Desktop/model_YOLOv4.pt")['state_dict'])
YOLO_v4_Module_WIP.eval()

#for current_batch_no in range(len(image_path_list) // batch_size):
for current_batch_no in range(0, 1200):
    input = image_reader(image_path_list[batch_size * current_batch_no: batch_size * (current_batch_no + 1)])
    input_tensor = torch.Tensor(input).cuda()
    #print(input.shape)
    output_137, output_148, output_159 = YOLO_v4_Module_WIP(input_tensor)
    print(output_137.shape)
    file_name = 'F:/FlyAI/UnderwaterDetection_roundB/TL_input_data/' + str(current_batch_no) + '.pt'
    torch.save({'output_137': output_137, 'output_148': output_148, 'output_159': output_159}, file_name)
    
"""
#torch.save({'output_137': output_137}, 'Model/output.pt')




#import training data
#convert image into label data (for training purpose)

round_A_data_folder = "F:/FlyAI/UnderwaterDetection_roundA/"
round_A_image_folder = round_A_data_folder + "train-A/image/"
round_A_box_folder = round_A_data_folder + "train-A/box/"

box_file_path_list = glob.glob(round_A_box_folder + "*.xml")
box_file_name_list = [x.split('\\')[1] for x in box_file_path_list]

image_file_name_list = [(x.split('.')[0] + '.jpg') for x in box_file_name_list]
image_file_path_list = [(round_A_image_folder + x) for x in image_file_name_list]


anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
anchor_shape_1 = int(len(anchors) / 2)
anchors = np.array(anchors).reshape(anchor_shape_1, 2)
def best_anchor(image_info):
    max_IoU = 0
    for m in range(9):
        x_GT = 0
        y_GT = 0
        w_GT = image_info[9]
        h_GT = image_info[10]
        x_PD = image_info[9] / 2 - anchors[m][0] / 2
        y_PD = image_info[10] / 2 - anchors[m][1] / 2
        w_PD = anchors[m][0]
        h_PD = anchors[m][1]
        
        current_IOU = IoU(x_GT, y_GT, w_GT, h_GT, x_PD, y_PD, w_PD, h_PD)
        #print(current_IOU)
        if current_IOU >= max_IoU:
            max_IoU = current_IOU
            selected_anchors = m
    
    if selected_anchors < 3:
        grid_size = 608 / 76
    elif selected_anchors < 6:
        grid_size = 608 / 38
    elif selected_anchors < 9:
        grid_size = 608 / 19
    
    grid_no_x = int((image_info[2] + image_info[4]) / 2 / grid_size)
    grid_no_y = int((image_info[3] + image_info[5]) / 2 / grid_size)
    
    w_expanded_time = image_info[9] / w_PD
    h_expanded_time = image_info[10] / h_PD
    
    position_x = (image_info[2] + image_info[4]) / 2 / grid_size - grid_no_x
    position_y = (image_info[3] + image_info[5]) / 2 / grid_size - grid_no_y
    
    #print(image_info)
    #print(grid_no_x)
    #print(grid_no_y)
    return selected_anchors, grid_no_x, grid_no_y, position_x, position_y, w_expanded_time, h_expanded_time

def read_xml_into_training_data(box_file_path_list):
    All_image = []
    for box_file_path in box_file_path_list:
        dict_object = {"holothurian": 0, "echinus": 1, "scallop": 2, "starfish": 3, 'waterweeds': 0}
        tree = ET.parse(box_file_path)
        root = tree.getroot()
        """
        root.tag
        root.attrib
        
        for child in root:
            for sub_child in child:
                print(sub_child.tag)
        """
        
        object = [None for i in range(6)]
        All_object = []
        All_image_size = []
        image_size = [None, None]
        frame_name = []
        image_size_width = []
        image_size_height = []
        object_name = []
        object_type = []
        object_xmin = []
        object_ymin = []
        object_xmax = []
        object_ymax = []

        for name in root.findall("./frame"):
            frame_name.append(name.text)
        for name in root.findall("./size/width"):
            image_size_width.append(int(name.text))
        for name in root.findall("./size/height"):
            image_size_height.append(int(name.text))

        for name in root.findall("./object/name"):
            object_name.append(name.text)

        for name in root.findall("./object/bndbox/xmin"):
            object_xmin.append(int(name.text))
        for name in root.findall("./object/bndbox/ymin"):
            object_ymin.append(int(name.text))
        for name in root.findall("./object/bndbox/xmax"):
            object_xmax.append(int(name.text))
        for name in root.findall("./object/bndbox/ymax"):
            object_ymax.append(int(name.text))

        for i in range(len(object_name)):
            current_object = []
            current_object.append(object_name[i])
            current_object.append(dict_object[object_name[i]])
            current_object.append(object_xmin[i])
            current_object.append(object_ymin[i])
            current_object.append(object_xmax[i])
            current_object.append(object_ymax[i])
            current_object.append(frame_name[0])
            current_object.append(image_size_width[0])
            current_object.append(image_size_height[0])
            All_object.append(current_object)
        All_image.append(All_object)
    return All_image


training_input_data = read_image(image_file_path_list[0: 2])

All_image = read_xml_into_training_data(box_file_path_list[0:4000])

yolo_size = 608

for i in range(len(All_image)):
    for j in range(len(All_image[i])):
        All_image[i][j][2] = All_image[i][j][2] / All_image[i][j][7] * yolo_size
        All_image[i][j][4] = All_image[i][j][4] / All_image[i][j][7] * yolo_size
        All_image[i][j][3] = All_image[i][j][3] / All_image[i][j][8] * yolo_size
        All_image[i][j][5] = All_image[i][j][5] / All_image[i][j][8] * yolo_size
        All_image[i][j].append(All_image[i][j][4] - All_image[i][j][2])
        All_image[i][j].append(All_image[i][j][5] - All_image[i][j][3])
        selected_anchors, grid_no_x, grid_no_y, position_x, position_y, w_expanded_time, h_expanded_time = best_anchor(All_image[i][j])
        All_image[i][j].append(selected_anchors)
        All_image[i][j].append(grid_no_x)
        All_image[i][j].append(grid_no_y)
        All_image[i][j].append(position_x)
        All_image[i][j].append(position_y)
        All_image[i][j].append(w_expanded_time)
        All_image[i][j].append(h_expanded_time)
        
All_image[0][0]


@profile
def TL_model_training():
    learning_rate = 0.001
    epoch_size = 1
    batch_size = 50
    steps_for_printing_out_loss = 1
    
    loss_function_MSE = nn.MSELoss(size_average=False)
    loss_function_BCE = nn.BCELoss(size_average=False)
    
    #loss_function_MSE = nn.MSELoss()
    #loss_function_BCE = nn.BCELoss()
    
    TL_model = transfer_learning_model().cuda()
    optimizer = optim.SGD(TL_model.parameters(), lr = learning_rate)
    #TL_model.load_state_dict(torch.load('Model/TL_model_starting_point.pt')['state_dict'])
    #TL_model.load_state_dict(torch.load('Model/TL_model1.pt')['state_dict'])
    #TL_model.eval()
    
    for i in range(1, epoch_size + 1):
        #loss = 0
        for current_batch_no in range(int(100/ batch_size)):
            optimizer.zero_grad()
            total_loss = 0
            total_loss_w_h = 0
            total_loss_class = 0
            total_loss_obj_p = 0
            total_loss_x_y = 0
            for image_pt_name in range(current_batch_no * batch_size, (current_batch_no + 1) * batch_size):
                torch.cuda.empty_cache()
                if image_pt_name % 200 == 0:
                    print(image_pt_name)

                file_name = 'F:/FlyAI/TL_input_data/' + str(image_pt_name) + '.pt'
                input_data = torch.load(file_name)
                layer_137_out = input_data['output_137'].cuda()
                layer_148_out = input_data['output_148'].cuda()
                layer_159_out = input_data['output_159'].cuda()
                #print(layer_137_out.shape)
                
                output_76, output_38, output_19 = TL_model(layer_137_out, layer_148_out, layer_159_out)
                output_file_name = 'F:/FlyAI/TL_output_data/' + str(image_pt_name) + '.pt'
                torch.save({'output_76': output_76, 'output_38': output_38, 'output_19': output_19}, output_file_name)
                
                target_76 = output_76.clone()
                target_38 = output_38.clone()
                target_19 = output_19.clone()
                #there is a possibility of more than one GT are mapped into same grid of a anchor, may need to check from training data?
                image_set = range(0, 1)
                target_76[:,:,4,:,:] = 0
                target_38[:,:,4,:,:] = 0
                target_19[:,:,4,:,:] = 0
                
                for image_no in image_set:
                    image_no_current_batch = image_no % batch_size

                    for item in All_image[image_pt_name]:
                        #print(item)
                        anchor_no = item[11] % 3
                        grid_x = item[12]
                        grid_y = item[13]

                        obj_class_no = item[1]
                        central_x = (item[2] + item[4]) / 2
                        central_y = (item[3] + item[5]) / 2
                        width_x = item[9]
                        height_y = item[10]

                        position_x = item[14]
                        position_y = item[15]
                        w_expanded_time = item[16]
                        h_expanded_time = item[17]

                        if item[11] < 3:
                            #target_76[anchor_no,image_no_current_batch,4,:,:] = 0
                            target_76[anchor_no,image_no_current_batch,0,grid_y,grid_x] = position_x
                            target_76[anchor_no,image_no_current_batch,1,grid_y,grid_x] = position_y
                            target_76[anchor_no,image_no_current_batch,2,grid_y,grid_x] = w_expanded_time
                            target_76[anchor_no,image_no_current_batch,3,grid_y,grid_x] = h_expanded_time
                            target_76[anchor_no,image_no_current_batch,4,grid_y,grid_x] = 1

                            target_76[anchor_no,image_no_current_batch,5:9,grid_y,grid_x] = 0
                            target_76[anchor_no,image_no_current_batch,5 + obj_class_no,grid_y,grid_x] = 1
                        elif item[11] < 6:
                            #target_38[anchor_no,image_no_current_batch,4,:,:] = 0
                            target_38[anchor_no,image_no_current_batch,0,grid_y,grid_x] = position_x
                            target_38[anchor_no,image_no_current_batch,1,grid_y,grid_x] = position_y
                            target_38[anchor_no,image_no_current_batch,2,grid_y,grid_x] = w_expanded_time
                            target_38[anchor_no,image_no_current_batch,3,grid_y,grid_x] = h_expanded_time
                            target_38[anchor_no,image_no_current_batch,4,grid_y,grid_x] = 1

                            target_38[anchor_no,image_no_current_batch,5:9,grid_y,grid_x] = 0
                            target_38[anchor_no,image_no_current_batch,5 + obj_class_no,grid_y,grid_x] = 1
                        elif item[11] < 9:
                            #target_19[anchor_no,image_no_current_batch,4,:,:] = 0
                            target_19[anchor_no,image_no_current_batch,0,grid_y,grid_x] = position_x
                            target_19[anchor_no,image_no_current_batch,1,grid_y,grid_x] = position_y
                            target_19[anchor_no,image_no_current_batch,2,grid_y,grid_x] = w_expanded_time
                            target_19[anchor_no,image_no_current_batch,3,grid_y,grid_x] = h_expanded_time
                            target_19[anchor_no,image_no_current_batch,4,grid_y,grid_x] = 1

                            target_19[anchor_no,image_no_current_batch,5:9,grid_y,grid_x] = 0
                            target_19[anchor_no,image_no_current_batch,5 + obj_class_no,grid_y,grid_x] = 1

                #global output_tensor
                #output_tensor = output
                #print(output_76.shape)
                #print(output_76)

                target_file_name = 'F:/FlyAI/TL_output_data/target_' + str(image_pt_name) + '.pt'
                torch.save({'target_76': target_76, 'target_38': target_38, 'target_19': target_19}, target_file_name)



                target_76 = Variable(target_76, requires_grad=False)
                target_38 = Variable(target_38, requires_grad=False)
                target_19 = Variable(target_19, requires_grad=False)

                #print(target_76[0])

                loss_w_h = loss_function_MSE(output_76[:, :, 2 : 4, :, :], target_76[:, :, 2 : 4, :, :])\
                 + loss_function_MSE(output_38[:, :, 2 : 4, :, :], target_38[:, :, 2 : 4, :, :])\
                 + loss_function_MSE(output_19[:, :, 2 : 4, :, :], target_19[:, :, 2 : 4, :, :])

                loss_w_h = loss_w_h / 2

                loss_class = loss_function_BCE(output_76[:, :, 5 : 9, :, :], target_76[:, :, 5 : 9, :, :])\
                 + loss_function_BCE(output_38[:, :, 5 : 9, :, :], target_38[:, :, 5 : 9, :, :])\
                 + loss_function_BCE(output_19[:, :, 5 : 9, :, :], target_19[:, :, 5 : 9, :, :])

                loss_obj_p = loss_function_BCE(output_76[:, :, 4, :, :], target_76[:, :, 4, :, :])\
                 + loss_function_BCE(output_38[:, :, 4, :, :], target_38[:, :, 4, :, :])\
                 + loss_function_BCE(output_19[:, :, 4, :, :], target_19[:, :, 4, :, :])

                loss_x_y = loss_function_BCE(output_76[:, :, 0 : 2, :, :], target_76[:, :, 0 : 2, :, :])\
                 + loss_function_BCE(output_38[:, :, 0 : 2, :, :], target_38[:, :, 0 : 2, :, :])\
                 + loss_function_BCE(output_19[:, :, 0 : 2, :, :], target_19[:, :, 0 : 2, :, :])

                loss = loss_w_h + loss_class + loss_obj_p + loss_x_y
                #loss = loss_function_MSE(output_76, target_76) + loss_function_MSE(output_38, target_38) + loss_function_MSE(output_19, target_19)

                total_loss += loss
                total_loss_w_h += loss_w_h
                total_loss_class += loss_class
                total_loss_obj_p += loss_obj_p
                total_loss_x_y += loss_x_y
                gc.collect()
                print('Max memory allocated: {0:.2f} MB'
                      .format(torch.cuda.max_memory_allocated() / 1e6))
                print('Max memory cached: {0:.2f} MB'
                      .format(torch.cuda.max_memory_cached() / 1e6))
                loss.backward()

            #loss = loss_function(output_76, target_76)

            optimizer.step()
            if i % (steps_for_printing_out_loss) == 0:
                print('Loss (epoch: ' + str(i) + '): ' + str(total_loss.cpu().detach().numpy()))
                print(total_loss_w_h)
                print(total_loss_class)
                print(total_loss_obj_p)
                print(total_loss_x_y)
                TL_model_file_path_by_epoch = "Model/TL_model"+ str(i) + ".pt" 
                torch.save({'state_dict': TL_model.state_dict(),'optimizer': optimizer.state_dict()}, TL_model_file_path_by_epoch)
            
    torch.save({'state_dict': TL_model.state_dict(),'optimizer': optimizer.state_dict()}, TL_model_file_path)

TL_model_training()