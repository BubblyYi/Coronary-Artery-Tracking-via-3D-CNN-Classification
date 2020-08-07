# -*- coding: UTF-8 -*-
# @Time    : 12/05/2020 20:06
# @Author  : BubblyYi
# @FileName: patch_generater.py
# @Software: PyCharm

import SimpleITK as sitk
import numpy as np
np.random.seed(4)
import pandas as pd
import os
from utils import resample, get_spacing_res2, get_start_ind, get_end_ind, get_new_radial_ind, get_shell, get_pre_next_point_ind, rotate_augmentation, find_closer_point_angle


def creat_data(max_points,path_name,spacing_path,gap_size,save_num):
    '''

    :param max_points:
    :param path_name:
    :param spacing_path:
    :param gap_size:
    :param save_num:
    :return:
    '''
    spacing_info = np.loadtxt(spacing_path,
                              delimiter=",", dtype=np.float32)
    pre_ind_list = []
    next_ind_list = []
    radials_list = []
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
                                                           np.array([0.5, 0.5, 0.5]))

    curr_mean = np.array([0, 0, 0])

    rotate_prob = 0.3

    for v in range(4):
    # for v in range(1):
        print("processing vessel %d" % v)
        reference_path = path_name + '0' + str(i) + '/' + 'vessel' + str(v) + '/' + 'reference.txt'
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        center = txt_data[..., 0:3]

        radials_data = txt_data[..., 3]
        start_ind = get_start_ind(center, radials_data)
        end_ind = get_end_ind(center, radials_data)
        print("start ind:", start_ind)
        print("end ind:", end_ind)
        counter = 0
        last_center_x_pixel = -1
        last_center_y_pixel = -1
        last_center_z_pixel = -1
        # for j in range(start_ind, start_ind+1):
        for j in range(start_ind, end_ind + 1):
            if j % gap_size == 0:
                print('j:', j)
                center_x = center[j][0]
                center_y = center[j][1]
                center_z = center[j][2]

                org_x_pixel = get_spacing_res2(center_x, spacing_x, resize_factor[1])
                org_y_pixel = get_spacing_res2(center_y, spacing_y, resize_factor[2])
                org_z_pixel = get_spacing_res2(center_z, spacing_z, resize_factor[0])

                if org_x_pixel!=last_center_x_pixel or org_y_pixel!=last_center_y_pixel or org_z_pixel!=last_center_z_pixel:
                    print("last:",[last_center_x_pixel,last_center_y_pixel,last_center_z_pixel])
                    print("curr:",[org_x_pixel, org_y_pixel, org_z_pixel])
                    last_center_x_pixel = org_x_pixel
                    last_center_y_pixel = org_y_pixel
                    last_center_z_pixel = org_z_pixel

                    radial = radials_data[j]

                    record_set = set()
                    curr_conv = np.array([[radial * 0.25, 0.0, 0.0],
                                          [0.0, radial * 0.25, 0.0],
                                          [0.0, 0.0, radial * 0.25]])

                    # To then obtain an off-centerline sample, point x is translated using a random shift sampled from a 3D normal distribution with μ = 0.0, σ = 0.25r

                    for k in range(10):
                        off_center_x, off_center_y, off_center_z = np.random.multivariate_normal(mean=curr_mean,
                                                                                                 cov=curr_conv,
                                                                                                 size=1).T
                        center_x_new = center_x + off_center_x[0]
                        center_y_new = center_y + off_center_y[0]
                        center_z_new = center_z + off_center_z[0]
                        center_x_pixel = get_spacing_res2(center_x_new, spacing_x, resize_factor[1])
                        center_y_pixel = get_spacing_res2(center_y_new, spacing_y, resize_factor[2])
                        center_z_pixel = get_spacing_res2(center_z_new, spacing_z, resize_factor[0])

                        while True:
                            if (center_x_pixel != org_x_pixel or center_y_pixel != org_y_pixel or center_z_pixel != org_z_pixel) and (center_x_pixel, center_y_pixel, center_z_pixel) not in record_set:
                                record_set.add((center_x_pixel, center_y_pixel, center_z_pixel))
                                break
                            else:
                                off_center_x, off_center_y, off_center_z = np.random.multivariate_normal(mean=curr_mean,
                                                                                                         cov=curr_conv,
                                                                                                         size=1).T
                                center_x_new = center_x + off_center_x[0]
                                center_y_new = center_y + off_center_y[0]
                                center_z_new = center_z + off_center_z[0]

                                center_x_pixel = get_spacing_res2(center_x_new, spacing_x, resize_factor[1])
                                center_y_pixel = get_spacing_res2(center_y_new, spacing_y, resize_factor[2])
                                center_z_pixel = get_spacing_res2(center_z_new, spacing_z, resize_factor[0])


                        new_radial_ind = get_new_radial_ind(center,[center_x_new, center_y_new, center_z_new])

                        new_radial = radials_data[new_radial_ind]

                        sx, sy, sz = get_shell(max_points, new_radial)
                        shell_arr = np.zeros((len(sx), 3))
                        for s_ind in range(len(sx)):
                            shell_arr[s_ind][0] = sx[s_ind]
                            shell_arr[s_ind][1] = sy[s_ind]
                            shell_arr[s_ind][2] = sz[s_ind]

                        pre_ind, next_ind = get_pre_next_point_ind(center, radials_data, new_radial_ind)
                        # 只有找到了前一个点和后一个点才进入切割流程
                        if pre_ind != -1 and next_ind != -1:


                            cut_size = 9

                            left_x = center_x_pixel - cut_size
                            right_x = center_x_pixel + cut_size
                            left_y = center_y_pixel - cut_size
                            right_y = center_y_pixel + cut_size
                            left_z = center_z_pixel - cut_size
                            right_z = center_z_pixel + cut_size

                            new_src_arr = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                            for ind in range(left_z, right_z + 1):
                                src_temp = re_spacing_img[ind].copy()
                                new_src_arr[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]

                            if np.random.uniform() <= rotate_prob:
                                curr_c = [center_x_new, center_y_new, center_z_new]
                                new_src_arr,new_pre_cood, new_next_cood= rotate_augmentation(new_src_arr,
                                                                                             pre_ind,
                                                                                             next_ind,
                                                                                             curr_c,
                                                                                             center,
                                                                                             angle_x = (-60. / 360 * 2. * np.pi, 60. / 360 * 2. * np.pi),
                                                                                             angle_y = (-60. / 360 * 2. * np.pi, 60. / 360 * 2. * np.pi),
                                                                                             angle_z =(-60. / 360 * 2. * np.pi, 60. / 360 * 2. * np.pi))
                                p = [new_pre_cood[0], new_pre_cood[1], new_pre_cood[2]]
                                pre_sim = find_closer_point_angle(shell_arr, p, curr_c)
                                p = [new_next_cood[0], new_next_cood[1], new_next_cood[2]]
                                next_sim = find_closer_point_angle(shell_arr, p, curr_c)
                                pre_ind_list.append(pre_sim)
                                next_ind_list.append(next_sim)
                                radials_list.append(new_radial)
                            else:
                                pre_x = center[pre_ind][0]
                                pre_y = center[pre_ind][1]
                                pre_z = center[pre_ind][2]

                                next_x = center[next_ind][0]
                                next_y = center[next_ind][1]
                                next_z = center[next_ind][2]

                                curr_c = [center_x_new, center_y_new, center_z_new]
                                p = [pre_x, pre_y, pre_z]
                                pre_sim = find_closer_point_angle(shell_arr, p, curr_c)
                                p = [next_x, next_y, next_z]
                                next_sim = find_closer_point_angle(shell_arr, p, curr_c)
                                pre_ind_list.append(pre_sim)
                                next_ind_list.append(next_sim)
                                radials_list.append(new_radial)
                            folder_path = './patch_data/centerline_patch/offset/point_' + str(max_points) + '_gp_' + str(gap_size)+'/'+'d'+str(i)
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                            record_name = 'centerline_patch/offset/point_' + str(max_points) + '_gp_' + str(gap_size)+'/'+'d'+str(i)+'/' + 'd_' + str(i) + '_' + 'v_' + str(v) + '_' + 'patch_%d_' % counter+str(k)+'.nii.gz'
                            org_name = './patch_data/' + record_name
                            out = sitk.GetImageFromArray(new_src_arr)
                            sitk.WriteImage(out, org_name)
                            patch_name.append(record_name)

                    counter += 1

    return pre_ind_list, next_ind_list, radials_list, patch_name

def create_patch_images(max_points,path_name,spacing_path,gap_size):
    # for i in range(1):
    for i in range(8):
        pre_ind_list, next_ind_list, radials_list, patch_name = creat_data(max_points,path_name,spacing_path,gap_size,i)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'pre_ind': pre_ind_list, 'next_ind': next_ind_list, 'radials': radials_list})
        print(dataframe.head())
        csv_name = "./patch_data/centerline_patch/offset/" + 'point_' + str(max_points) + '_gp_' + str(
            gap_size)+'/'+'d'+str(i) + "_patch_info_%d.csv" % max_points
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'pre_ind', 'next_ind', 'radials'], sep=',')
        print("create patch info csv")
        print("down")

max_points = 500
gap_size = 1
path_name = 'train_data/dataset'
spacing_path = 'spacing_info.csv'

create_patch_images(max_points,path_name,spacing_path,gap_size)