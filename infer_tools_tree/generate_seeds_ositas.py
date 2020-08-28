# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 18:55
# @Author  : BubblyYi、QYD
# @FileName: centerline_net.py
# @Software: PyCharm
import numpy as np
from setting import src_array, spacing, seeds_model, ostia_model, device
from utils import data_preprocess, resample, crop_heart
import torch

def search_seeds_ostias(max_size=(200, 10)):
    '''
    find seeds points arr and ostia points arr
    :param max_size: The first max_size[0] seed points and the first max_size[1] ostia points were selected
    :return:
    '''
    print("search seeds and ostias")
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    re_spacing_img, curr_spacing, resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))
    re_spacing_img, meam_minc, mean_minr, mean_maxc, mean_maxr = crop_heart(re_spacing_img)
    cut_size = 9
    res_seeds = {}
    res_ostia = {}
    count = 0
    random_point_size = 80000
    batch_size = 1000
    new_patch_list = []
    center_coord_list = []
    z, h, w = re_spacing_img.shape
    offset_size = 10
    x_list = np.random.random_integers(meam_minc - offset_size, mean_maxc + offset_size, (random_point_size, 1))
    y_list = np.random.random_integers(mean_minr - offset_size, mean_maxr + offset_size, (random_point_size, 1))
    z_list = np.random.random_integers(0, z, (random_point_size, 1))

    index = np.concatenate([x_list, y_list, z_list], axis=1)
    index = list(set(tuple(x) for x in index))
    for i in index:
        center_x_pixel = i[0]
        center_y_pixel = i[1]
        center_z_pixel = i[2]
        left_x = center_x_pixel - cut_size
        right_x = center_x_pixel + cut_size
        left_y = center_y_pixel - cut_size
        right_y = center_y_pixel + cut_size
        left_z = center_z_pixel - cut_size
        right_z = center_z_pixel + cut_size
        if left_x >= 0 and right_x < h and left_y >= 0 and right_y < w and left_z >= 0 and right_z < z:
            new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
            for ind in range(left_z, right_z + 1):
                src_temp = re_spacing_img[ind].copy()
                new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
            count += 1
            input_data = data_preprocess(new_patch)
            new_patch_list.append(input_data)
            center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))
            if count % batch_size == 0:
                input_data = torch.cat(new_patch_list, axis=0)
                inputs = input_data.to(device)
                seeds_outputs = seeds_model(inputs.float())
                seeds_outputs = seeds_outputs.view((len(input_data)))  # view
                seeds_proximity = seeds_outputs.cpu().detach().numpy()
                ostia_outputs = ostia_model(inputs.float())
                ostia_outputs = ostia_outputs.view(len(input_data))
                ostia_proximity = ostia_outputs.cpu().detach().numpy()
                for i in range(batch_size):
                    res_seeds[center_coord_list[i]] = seeds_proximity[i]
                    res_ostia[center_coord_list[i]] = ostia_proximity[i]
                new_patch_list.clear()
                center_coord_list.clear()
                del input_data
                del inputs
                del seeds_outputs
                del ostia_outputs

    positive_count = 0
    for i in res_seeds.values():
        if i > 0:
            positive_count += 1
    res_seeds = sorted(res_seeds.items(), key=lambda item: item[1], reverse=True)
    res_ostia = sorted(res_ostia.items(), key=lambda item: item[1], reverse=True)
    res_seeds = res_seeds[:max_size[0]]
    res_ostia = res_ostia[:max_size[1]]
    return res_seeds, res_ostia