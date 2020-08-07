# -*- coding: UTF-8 -*-
# @Time    : 12/05/2020 20:06
# @Author  : BubblyYi
# @FileName: patch_generater.py
# @Software: PyCharm

import SimpleITK as sitk
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import math
from scipy.ndimage.interpolation import zoom
import warnings
import os
np.random.seed(4)
from utils import resample, get_shell, get_proximity, get_closer_distence

def creat_data(path_name,spacing_path,gap_size,save_num):
    spacing_info = np.loadtxt(spacing_path,
                              delimiter=",", dtype=np.float32)
    proximity_list = []
    patch_name = []
    i = save_num
    print("processing dataset %d" % i)
    image_pre_fix = path_name + '0' + str(i) + '/' + 'image' + '0' + str(i)
    file_name = image_pre_fix + '.nii.gz'
    src_array = sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32))

    spacing_x = spacing_info[i][0]
    spacing_y = spacing_info[i][1]
    spacing_z = spacing_info[i][2]
    re_spacing_img, curr_spacing, resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))

    max_z, max_y, max_x = re_spacing_img.shape

    vessels = []
    for j in range(4):
        reference_path = '/data_process_tools/train_data/dataset0'+str(i)+'/vessel' + str(j) + '/reference.txt'
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        temp_center = txt_data[..., 0:3]
        vessels.append(temp_center)

    record_set = set()
    max_range = 4
    max_points = 30
    for v in range(4):
        print("processing vessel %d" % v)
        reference_path = path_name + '0' + str(i) + '/' + 'vessel' + str(v) + '/' + 'reference.txt'
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        center = txt_data[..., 0:3]

        counter = 0

        last_center_x_pixel = -1
        last_center_y_pixel = -1
        last_center_z_pixel = -1

        for j in range(len(center)):
            center_x = center[j][0]
            center_y = center[j][1]
            center_z = center[j][2]
            record_set.add((center_x, center_y, center_z))
            if j % gap_size == 0:

                org_x_pixel = int(round(center_x))
                org_y_pixel = int(round(center_y))
                org_z_pixel = int(round(center_z))
                record_set.add((org_x_pixel,org_y_pixel,org_z_pixel))
                if org_x_pixel!=last_center_x_pixel or org_y_pixel!=last_center_y_pixel or org_z_pixel!=last_center_z_pixel:
                    last_center_x_pixel = org_x_pixel
                    last_center_y_pixel = org_y_pixel
                    last_center_z_pixel = org_z_pixel
                    for k in range(1,max_range+1):
                        x_list,y_list,z_list = get_shell(max_points,k)
                        for m in range(len(x_list)):
                            new_x = int(round(center_x + x_list[m]))
                            new_y = int(round(center_y + y_list[m]))
                            new_z = int(round(center_z + z_list[m]))
                            check_temp = (new_x,new_y,new_z)
                            if check_temp not in record_set:
                                record_set.add(check_temp)
                                center_x_pixel = new_x
                                center_y_pixel = new_y
                                center_z_pixel = new_z

                                target_point = np.array([center_x_pixel, center_y_pixel, center_z_pixel])
                                print("new center:", target_point)
                                min_dis = get_closer_distence(vessels, target_point)
                                curr_proximity = get_proximity(min_dis,cutoff_value=4)
                                print('proximity:', curr_proximity)
                                cut_size = 9

                                left_x = center_x_pixel - cut_size
                                right_x = center_x_pixel + cut_size
                                left_y = center_y_pixel - cut_size
                                right_y = center_y_pixel + cut_size
                                left_z = center_z_pixel - cut_size
                                right_z = center_z_pixel + cut_size

                                if (right_z + 1) < len(re_spacing_img) and left_z >= 0 and (right_y + 1) < max_y and left_y >= 0 and (right_x + 1) < max_x and left_x >= 0 and curr_proximity>0:
                                    new_src_arr = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                                    for ind in range(left_z, right_z + 1):
                                        src_temp = re_spacing_img[ind].copy()
                                        new_src_arr[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]

                                    folder_path = './patch_data/seeds_patch/positive/' + 'gp_' + str(gap_size) + '/d' + str(i)
                                    if not os.path.exists(folder_path):
                                        os.makedirs(folder_path)
                                    record_name = 'seeds_patch/positive/' + 'gp_' + str(gap_size) + '/d' + str(i) + '/' + 'd_' + str(i) + '_v_' +str(v)+ '_x_' + str(center_x_pixel) + '_y_' + str(center_y_pixel) + '_z_' + str(center_z_pixel) + '.nii.gz'
                                    print(record_name)
                                    org_name = './patch_data/' + record_name
                                    out = sitk.GetImageFromArray(new_src_arr)
                                    sitk.WriteImage(out, org_name)
                                    proximity_list.append(curr_proximity)
                                    patch_name.append(record_name)
                                    counter += 1

                                else:
                                    print('out of bounder skip this block')
            # break

    return patch_name, proximity_list

def create_patch_images(path_name,spacing_path,gap_size):
    for i in range(8):
        patch_name,proximity_list = creat_data(path_name,spacing_path,gap_size,i)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'proximity': proximity_list})
        print(dataframe.head())
        csv_name = "./patch_data/seeds_patch/positive/"+ 'gp_' + str(gap_size)+'/'+'d'+str(i) + "_patch_info.csv"
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'proximity'], sep=',')
        print("create patch info csv")
        print("down")

path_name = 'train_data/dataset'
spacing_path = 'spacing_info.csv'
gap_size  = 100
create_patch_images(path_name,spacing_path,gap_size)