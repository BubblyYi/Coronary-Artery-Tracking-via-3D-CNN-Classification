# -*- coding: UTF-8 -*-
# @Time    : 04/02/2020 10:58
# @Author  : BubblyYi
# @FileName: ostia_net_data_provider_aug.py
# @Software: PyCharm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import random
from scipy.ndimage import map_coordinates

class DataGenerater(Dataset):

    def __init__(self,data_path, pre_fix_path, transform = None, flag = '', target_transform = None):
        self.flag = flag
        data = []
        print("csv path:",data_path)
        csv_data = pd.read_csv(data_path)
        x_data = csv_data['patch_name']
        if self.flag == 'train' or self.flag == 'val':
            proximity = csv_data["proximity"]

            for i in range(len(x_data)):
                if pre_fix_path is None:
                    data.append((temp, proximity[i]))
                else:
                    temp = os.path.join(pre_fix_path,x_data[i])
                    data.append((temp, proximity[i]))
        else:
            for i in range(len(x_data)):
                if pre_fix_path is None:
                    data.append(x_data[i])
                else:
                    temp = os.path.join(pre_fix_path, x_data[i])
                    data.append(temp)

        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.p_gaussian_noise = 0.3
        self.rotate_prob = 0.3

    def __getitem__(self, index):
        if self.flag == 'train' or self.flag == 'val':
            data_path, p = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            proximity = p
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            img = (img - mean_intensity) / (std_intensity+1e-9)
            img = img.astype(np.float32)
            img = torch.from_numpy(img)


            return img.unsqueeze(0), proximity

        elif self.flag == 'test':
            data_path = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            # 防止除0
            img = (img - mean_intensity) / (std_intensity+1e-9)
            img = torch.from_numpy(img)
            return img.unsqueeze(0)

    def augment_gaussian_noise(self,data_sample, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:

            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)

        return data_sample

    def create_matrix_rotation_x_3d(self,angle, matrix=None):
        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(angle), -np.sin(angle)],
                               [0, np.sin(angle), np.cos(angle)]])
        if matrix is None:
            return rotation_x
        return np.dot(matrix, rotation_x)

    def create_matrix_rotation_y_3d(self,angle, matrix=None):
        rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                               [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]])
        if matrix is None:
            return rotation_y

        return np.dot(matrix, rotation_y)

    def create_matrix_rotation_z_3d(self,angle, matrix=None):
        rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0],
                               [0, 0, 1]])
        if matrix is None:
            return rotation_z

        return np.dot(matrix, rotation_z)

    def rotate_coords_3d(self,coords, angle_x, angle_y, angle_z):
        rot_matrix = np.identity(len(coords))
        rot_matrix = self.create_matrix_rotation_x_3d(angle_x, rot_matrix)
        rot_matrix = self.create_matrix_rotation_y_3d(angle_y, rot_matrix)
        rot_matrix = self.create_matrix_rotation_z_3d(angle_z, rot_matrix)
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        return coords

    def create_zero_centered_coordinate_mesh(self,shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    def interpolate_img(self,img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
        if is_seg and order != 0:
            unique_labels = np.unique(img)
            result = np.zeros(coords.shape[1:], img.dtype)
            for i, c in enumerate(unique_labels):
                res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
                result[res_new >= 0.5] = c
            return result
        else:
            return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


    def rotate_augmentation(self,data, p_rot_per_axis=1, angle_x=(0, 2 * np.pi),
                            angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi), border_mode_data='constant',
                            border_cval_data=0.0, order_data=3):
        patch_size = data.shape
        dim = 3
        # for sample_id in range(data.shape[0]):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
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
        coords = self.rotate_coords_3d(coords, a_x, a_y, a_z)

        modified_coords = True

        if modified_coords:
            for d in range(dim):
                ctr = int(np.round(data.shape[d] / 2.))
                coords[d] += ctr
            data_result = self.interpolate_img(data, coords, order_data, border_mode_data, cval=border_cval_data)

        return data_result

    def __len__(self):
        return len(self.data)