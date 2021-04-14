# -*- coding: UTF-8 -*-
# @Time    : 12/05/2020 20:06
# @Author  : BubblyYi
# @FileName: patch_generater.py
# @Software: PyCharm

import SimpleITK as sitk
import matplotlib
matplotlib.use('AGG')
import numpy as np
import pandas as pd
import os
np.random.seed(4)
from utils import resample, get_shell, get_proximity, get_closer_distence

def creat_data(path_name,spacing_path,save_num,cut_size = 19,move_step = 3):
    spacing_info = np.loadtxt(spacing_path, delimiter=",", dtype=np.float32)
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
    re_spacing_img, curr_spacing, resize_factor = resample(src_array,
                                                           np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))
    vessels = []
    for j in range(4):
        reference_path = './train_data/dataset0'+str(i)+'/vessel' + str(j) + '/reference.txt'
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        center = txt_data[..., 0:3]
        vessels.append(center)
    z, h, w = re_spacing_img.shape

    for iz in range(int((z - cut_size) / move_step + 1)):
        for ih in range(int((h - cut_size) / move_step + 1)):
            for iw in range(int((w - cut_size) / move_step + 1)):
                sz = iz * move_step
                ez = iz * move_step+cut_size

                sh = ih * move_step
                eh = ih * move_step+cut_size

                sw = iw * move_step
                ew = iw * move_step+cut_size
                center_z = (ez - sz) // 2 + sz
                center_y = (eh - sh) // 2 + sh
                center_x = (ew - sw) // 2 + sw
                target_point = np.array([center_x,center_y,center_z])
                print("new center:",target_point)
                min_dis = get_closer_distence(vessels, target_point)
                print('min dis:',min_dis)
                curr_proximity = get_proximity(min_dis)
                print('proximity:',curr_proximity)
                if curr_proximity<=0.0:
                    proximity_list.append(curr_proximity)
                    new_src_arr = np.zeros((cut_size, cut_size, cut_size))
                    for ind in range(sz, ez):
                        src_temp = re_spacing_img[ind].copy()
                        new_src_arr[ind - sz] = src_temp[sh:eh, sw:ew]

                    folder_path = './patch_data/seeds_patch/negative/'+'gp_' + str(move_step)+'/d'+str(i)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    record_name = 'seeds_patch/negative/' + 'gp_' + str(move_step)+'/d'+str(i)+'/' + 'd_' + str(i) + '_' + 'x_' + str(center_x) + '_y_'+str(center_y)+'_z_'+str(center_z)+'.nii.gz'
                    # print(record_name)
                    org_name = './patch_data/' + record_name
                    out = sitk.GetImageFromArray(new_src_arr)
                    sitk.WriteImage(out, org_name)
                    patch_name.append(record_name)

    return patch_name, proximity_list

def create_patch_images(path_name,spacing_path,cut_size = 19,move_step = 19):
    for i in range(8):
        patch_name,proximity_list = creat_data(path_name,spacing_path,i,cut_size,move_step)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'proximity': proximity_list})
        print(dataframe.head())
        csv_name = "./patch_data/seeds_patch/negative/"+ 'gp_' + str(
            move_step)+'/'+'d'+str(i) + "_patch_info.csv"
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'proximity'], sep=',')
        print("create patch info csv")
        print("down")

path_name = 'train_data/dataset'
spacing_path = 'spacing_info.csv'

create_patch_images(path_name,spacing_path,cut_size = 19,move_step = 19)