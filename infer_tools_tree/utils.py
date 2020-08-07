import numpy as np
import warnings
from scipy.ndimage.interpolation import zoom
import torch
import math
import copy
import cv2
from skimage import measure
import pandas as pd

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)

        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing, resize_factor
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def get_start_ind(center_points):
    curr_x = center_points[0][0]
    curr_y = center_points[0][1]
    curr_z = center_points[0][2]

    curr_r = 3
    start_ind = -1
    ellipsis = 0.1
    for i in range(1, len(center_points)):
        v1 = np.array([curr_x, curr_y, curr_z])
        v2 = np.array([center_points[i][0], center_points[i][1], center_points[i][2]])
        dist = np.linalg.norm(v1 - v2)
        if (dist - curr_r) <= ellipsis and dist >= curr_r:
            start_ind = i
            break
    return start_ind


def get_spacing_res2(x, spacing_x, spacing_new):
    return int(round((x / spacing_x) * spacing_new))


def get_world_cood(x, spacing_x, spacing_new):
    return (x / spacing_new) * spacing_x


def data_preprocess(img):
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    upper_bound = np.percentile(img, 99.5)
    lower_bound = np.percentile(img, 00.5)
    img = np.clip(img, lower_bound, upper_bound)
    # 防止除0
    img = (img - mean_intensity) / (std_intensity + 1e-9)
    img = np.array([img])
    img = torch.from_numpy(img)
    return img.unsqueeze(0)


def get_shell(fl_Num_Points, fl_Radius):
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
    return x_list, y_list, z_list


def prob_terminates(pre_y, max_points):

    res = torch.sum(-pre_y * torch.log2(pre_y))
    return res / torch.log2(torch.from_numpy(np.array([max_points])).float())


def get_closer_distance(vessel, target_point):
    min_dis = float("inf")
    for i in range(len(vessel)):
        curr_point = vessel[i]
        dist = np.linalg.norm(target_point - curr_point)
        if dist < min_dis:
            min_dis = dist
            index = i
    return min_dis, index


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def get_angle(v1, v2):
    cosangle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosangle = np.clip(cosangle, -1, 1)
    return math.degrees(np.arccos(cosangle))

def save_info(res: list, path: str):
    x_list = []
    y_list = []
    z_list = []
    for i in range(len(res)):
        x_list.append(res[i][0][0])
        y_list.append(res[i][0][1])
        z_list.append(res[i][0][2])
    dataframe = pd.DataFrame(
        {'x': x_list, 'y': y_list, 'z': z_list})
    dataframe.to_csv(path, index=False,
                     columns=['x', 'y', 'z'], sep=',',float_format='%.5f')

def crop_heart(input_arr):
    '''
    In order to remove the influence of pulmonary vessels, we will use threshold method to segment the heart region
    :param input_arr: image arr
    :return: Data after removing lung areas
    '''
    src_array = copy.deepcopy(input_arr)
    z, w, h = src_array.shape
    new_arr = np.zeros((z, w, h))
    new_arr += -1000
    sum_minr = 0
    sum_minc = 0
    sum_maxr = 0
    sum_maxc = 0
    for k in range(z):
        image = src_array[k][:, :]
        ret, thresh = cv2.threshold(image, 20, 400, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, anchor=(-1, -1), iterations=4)

        label_opening = measure.label(opening)
        regionprops = measure.regionprops(label_opening)

        max_area = 0
        index = 0
        for i in range(len(regionprops)):
            if regionprops[i].area > max_area:
                max_area = regionprops[i].area
                index = i
        minr, minc, maxr, maxc = regionprops[index].bbox
        new_arr[k][minr:maxr, minc:maxc] = src_array[k][minr:maxr, minc:maxc]
        sum_minr += minr
        sum_minc += minc
        sum_maxr += maxr
        sum_maxc += maxc
    mean_minr = sum_minr // z
    meam_minc = sum_minc // z
    mean_maxr = sum_maxr // z
    mean_maxc = sum_maxc // z

    return new_arr, meam_minc, mean_minr, mean_maxc, mean_maxr
