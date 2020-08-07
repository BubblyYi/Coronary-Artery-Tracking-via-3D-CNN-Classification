# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 15:38
# @Author  : BubblyYi
# @FileName: setting.py
# @Software: PyCharm
import sys
sys.path.append('..')
from models.centerline_net import CenterlineNet
from models.seedspoints_net import SeedspointsNet
from models.ostiapoints_net import OstiapointsNet
import SimpleITK as sitk
import numpy as np
import os
import yaml
from yaml import FullLoader
import torch
from utils import resample
print("start processing")

# Make sure that you have modified the settingy.yaml

with open("settingy.yaml", "r", encoding='utf-8') as f:
    tmp = f.read()
    setting_info = yaml.load(tmp, Loader=FullLoader)
os.environ["CUDA_VISIBLE_DEVICES"] = str(setting_info["gpuid"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path_seeds = setting_info["checkpoint_path_seeds"]
checkpoint_path_ostia = setting_info["checkpoint_path_ostia"]
checkpoint_path_infer = setting_info["checkpoint_path_infer"]
max_points = setting_info["max_points"]
file_name = setting_info["file_name"]
prob_thr = setting_info["prob_thr"]
max_size = setting_info["max_size"]
print("load net")
infer_model = CenterlineNet(n_classes=max_points)
checkpoint = torch.load(checkpoint_path_infer)
net_dict = checkpoint['net_dict']
infer_model.load_state_dict(net_dict)
infer_model.to(device)
infer_model.eval()

seeds_model = SeedspointsNet()
seeds_checkpoint = torch.load(checkpoint_path_seeds)['net_dict']
seeds_model.load_state_dict(seeds_checkpoint)
seeds_model.to(device)
seeds_model.eval()

ostia_model = OstiapointsNet()
ostia_checkpoint = torch.load(checkpoint_path_ostia)['net_dict']
ostia_model.load_state_dict(ostia_checkpoint)
ostia_model.to(device)
ostia_model.eval()
print("read image")
itkimage = sitk.ReadImage(file_name)
spacing = itkimage.GetSpacing()
src_array = sitk.GetArrayFromImage(itkimage)
spacing_x = spacing[0]
spacing_y = spacing[1]
spacing_z = spacing[2]
re_spacing_img, curr_spacing, resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                       np.array([0.5, 0.5, 0.5]))
print("create output folds")
if not os.path.exists(setting_info["seeds_gen_info_to_save"]):
    os.makedirs(setting_info["seeds_gen_info_to_save"])
if not os.path.exists(setting_info["ostias_gen_info_to_save"]):
    os.makedirs(setting_info["ostias_gen_info_to_save"])
if not os.path.exists(setting_info["infer_line_to_save"]):
    os.makedirs(setting_info["infer_line_to_save"])
if not os.path.exists(setting_info["fig_to_save"]):
    os.makedirs(setting_info["fig_to_save"])


