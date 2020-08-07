# -*- coding: UTF-8 -*-
# @Time    : 06/08/2020 14:05
# @Author  : BubblyYi
# @FileName: utils.py
# @Software: PyCharm

import numpy as np
import math
from scipy.ndimage.interpolation import zoom
import warnings
from scipy.ndimage import map_coordinates
import copy
np.random.seed(4)

def resample(imgs, spacing, new_spacing,order = 2):
    '''
    :param imgs: Original image arr
    :param spacing: sapcing of the original image
    :param new_spacing: new spacing
    :param order:zoom order
    :return:
    '''
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)

        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        # print("resize_factor:", resize_factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing, resize_factor
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def get_shell(fl_Num_Points,fl_Radius):

    '''
    create a shell
    :param fl_Num_Points: Number of points on the surface of spherical shell
    :param fl_Radius: Ball radius
    :return: shell point list
    '''

    x_list = []
    y_list = []
    z_list = []
    offset = 2.0 / fl_Num_Points
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(fl_Num_Points):
        z = ((i * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(1.0 - pow(z, 2.0))

        phi = ((i + 1) % fl_Num_Points) * increment

        x = math.cos(phi) * r
        y = math.sin(phi) * r
        x_list.append(fl_Radius * x)
        y_list.append(fl_Radius * y)
        z_list.append(fl_Radius * z)
    return x_list,y_list,z_list

def get_spacing_res(x,spacing_x):
    return int(round(x/spacing_x))

def get_spacing_res2(x,spacing_x,spacing_new):
    return int(round((x/spacing_x)*spacing_new))

def get_start_ind(center_points,radials_data):
    '''
    searching a point 3 mm from the entrance of the coronary artery
    :param center_points: center points arr
    :param radials_data: radials
    :return:index
    '''
    curr_x = center_points[0][0]
    curr_y = center_points[0][1]
    curr_z = center_points[0][2]
    curr_r = 3
    start_ind = -1
    ellipsis = 0.1
    for i in range(1,len(center_points)):
        v1 = np.array([curr_x, curr_y, curr_z])
        v2 = center_points[i]
        dist = np.linalg.norm(v1 - v2)
        if (dist-curr_r)<=ellipsis and dist>=curr_r:
            start_ind = i
            break
    return start_ind

def get_end_ind(center_points,radials_data):
    '''
    searching a point 3 mm from the end of the coronary artery
    :param center_points: center points arr
    :param radials_data: radials
    :return:
    '''
    curr_x = center_points[-1][0]
    curr_y = center_points[-1][1]
    curr_z = center_points[-1][2]
    curr_r = 3
    end_ind = -1
    ellipsis = 0.1
    for i in range(len(center_points)-2, -1, -1):
        v1 = np.array([curr_x, curr_y, curr_z])
        v2 = center_points[i]
        dist = np.linalg.norm(v1 - v2)
        if (dist-curr_r)<=ellipsis and dist>=curr_r:
            end_ind = i
            break
    return end_ind

def get_pre_next_point_ind(center_points,radials_data,center_ind):
    '''
    Find the previous point and the next point R from the current center point
    :param center_points: center points arr
    :param radials_data: radial data
    :param center_ind: target center
    :return: the index of previous point and the next point

    '''

    curr_x = center_points[center_ind][0]
    curr_y = center_points[center_ind][1]
    curr_z = center_points[center_ind][2]
    curr_r = radials_data[center_ind]

    pre_ind = -1
    next_ind = -1
    ellipsis = 0.1

    for i in range(center_ind-1,-1,-1):
        v1 = np.array([curr_x,curr_y,curr_z])
        v2 = center_points[i]
        dist = np.linalg.norm(v1 - v2)
        if (dist-curr_r)<=ellipsis and dist>=curr_r:
            pre_ind = i
            break

    for i in range(center_ind+1,len(center_points)):
        v1 = np.array([curr_x,curr_y,curr_z])
        v2 = center_points[i]
        dist = np.linalg.norm(v1 - v2)
        if (dist-curr_r)<=ellipsis and dist>=curr_r:
            next_ind = i
            break
    return pre_ind,next_ind

def get_angle(v1,v2):
    '''
    Calculate the angle between two vectors
    :param v1: 3d vector
    :param v2: 3d vector
    :return: angle
    '''
    cosangle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(np.arccos(cosangle))

def find_closer_point_angle(curr_shell_arr,p,center):
    '''
    Find the corresponding point of the vector with the smallest angle on the spherical shell
    :param curr_shell_arr: shell point arr
    :param p: target point
    :param center: center point
    :return: index of shell points
    '''
    p[0] = p[0] - center[0]
    p[1] = p[1] - center[1]
    p[2] = p[2] - center[2]
    angle_sim_dict = {}
    for i in range(len(curr_shell_arr)):
        shell_v = curr_shell_arr[i].copy()
        curr_sim = get_angle(np.array(shell_v),np.array(p))
        angle_sim_dict[i] = curr_sim
    # print(angle_sim_dict)
    min_ind = min(angle_sim_dict,key=angle_sim_dict.get)
    # print("degree:",angle_sim_dict[min_ind])
    return min_ind

def search_points_list(center_points,radials_data,start_ind,end_ind):

    curr_x = center_points[start_ind][0]
    curr_y = center_points[start_ind][1]
    curr_z = center_points[start_ind][2]
    curr_r = radials_data[start_ind]
    res = []
    ellipsis = 0.1
    for i in range(start_ind+1,end_ind+1):
        v1 = np.array([curr_x,curr_y,curr_z])
        v2 = center_points[i]

        dist = np.linalg.norm(v1 - v2)
        if (dist-curr_r)<=ellipsis and dist>=curr_r:
            res.append(i - 1)
            curr_x = center_points[i - 1][0]
            curr_y = center_points[i - 1][1]
            curr_z = center_points[i - 1][2]
            curr_r = radials_data[i - 1]
    return res

def get_new_radial_ind(center_points, new_center):

    v1 = np.array(new_center)
    dist_sim_dict = {}
    for i in range(len(center_points)):
        v2 = center_points[i]
        dist = np.linalg.norm(v1 - v2)
        dist_sim_dict[i] = dist

    min_ind = min(dist_sim_dict, key=dist_sim_dict.get)
    return min_ind


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x
    return np.dot(matrix, rotation_x)

def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)

def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)

def get_rotate_res(rotate_center,curr_cood,angle_x,angle_y,angle_z):

    temp_cood = copy.deepcopy(curr_cood)
    temp_cood[0] = temp_cood[0] - rotate_center[0]
    temp_cood[1] = temp_cood[1] - rotate_center[1]
    temp_cood[2] = temp_cood[2] - rotate_center[2]

    # SimpleITK保存图像是按照ZYX保存，因此对图像旋转时结果也按ZYX保存,所以这里直接旋转坐标时也要反着
    temp_cood = rotate_coords_3d(temp_cood, angle_z, angle_y, angle_x)

    temp_cood[0] = temp_cood[0] + rotate_center[0]
    temp_cood[1] = temp_cood[1] + rotate_center[1]
    temp_cood[2] = temp_cood[2] + rotate_center[2]
    # print('new:',curr_cood)
    return temp_cood


def rotate_augmentation(data,pre_ind,next_ind,rotate_center,center, p_rot_per_axis = 1,angle_x = (0, 2 * np.pi), angle_y = (0, 2 * np.pi), angle_z = (0, 2 * np.pi),border_mode_data = 'constant',border_cval_data = 0.0, order_data = 3):
    patch_size = data.shape
    dim = 3
    # for sample_id in range(data.shape[0]):
    coords = create_zero_centered_coordinate_mesh(patch_size)
    data_result = np.zeros((patch_size[0], patch_size[1], patch_size[2]),
                           dtype=np.float32)

    if np.random.uniform() <= p_rot_per_axis:
        a_x = np.random.uniform(angle_x[0], angle_x[1])
    else:
        a_x = 0

    if np.random.uniform() <= p_rot_per_axis:
        a_y = np.random.uniform(angle_y[0], angle_y[1])
    else:
        a_y = 0

    if np.random.uniform() <= p_rot_per_axis:
        a_z = np.random.uniform(angle_z[0], angle_z[1])
    else:
        a_z = 0
    coords = rotate_coords_3d(coords, a_x, a_y, a_z)

    modified_coords = True

    if modified_coords:
        for d in range(dim):
            ctr = int(np.round(data.shape[d] / 2.))
            coords[d] += ctr
        data_result = interpolate_img(data, coords, order_data,border_mode_data, cval=border_cval_data)

    vector_center = np.array(rotate_center)
    new_pre_cood = get_rotate_res(vector_center,center[pre_ind],a_x,a_y,a_z)
    new_next_cood = get_rotate_res(vector_center,center[next_ind],a_x,a_y,a_z)

    return data_result, new_pre_cood, new_next_cood

def get_proximity(min_dist, cutoff_value=4, alpha=6):
    if min_dist>cutoff_value:
        return 0
    else:
        return np.exp(alpha*(1-min_dist/cutoff_value))

def get_max_boundr(image_shape,target_point):
    min_dist = 2**31
    corner_list = [[0, 0, 0],
                   [image_shape[0], image_shape[1], image_shape[2]],
                   [0, image_shape[1], image_shape[2]],
                   [image_shape[0], 0, image_shape[2]],
                    [image_shape[0], image_shape[1], 0],
                    [image_shape[0], 0, 0],
                    [0, image_shape[1], 0],
                    [0, 0, image_shape[2]]]
    for c in corner_list:
        curr_point = np.array(c)
        dist = np.linalg.norm(target_point - curr_point)
        min_dist = min(min_dist,dist)
    return min_dist

def get_closer_distence(vessels,target_point):
    min_dis = 2**31
    closer_point = -1
    for vessel in vessels:
        for i in range(len(vessel)):
            curr_point = vessel[i]
            dist = np.linalg.norm(target_point - curr_point)
            if dist<min_dis:
                min_dis = dist
                closer_point = curr_point
    print('closer point:', closer_point)
    return min_dis