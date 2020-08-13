# -*- coding: UTF-8 -*-
# @Time    : 14/05/2020 17:56
# @Author  : BubblyYi
# @FileName: ostia_train_tools.py
# @Software: PyCharm
import sys
sys.path.append('..')
from models.ostiapoints_net import OstiapointsNet
from ostia_net_data_provider_aug import DataGenerater
from ostia_trainner import Trainer
import torch

def get_dataset(save_num=0):
    # Replace these paths to the path where you store the data
    train_data_info_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/ostia_patch/train_save_d"+str(save_num)+"_train.csv"
    train_pre_fix_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/ostia_patch"
    train_flag = 'train'
    train_transforms = None
    target_transform = None
    train_dataset = DataGenerater(train_data_info_path, train_pre_fix_path, train_transforms, train_flag, target_transform)

    val_data_info_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/ostia_patch/train_save_d"+str(save_num)+"_val.csv"
    val_pre_fix_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/ostia_patch"
    val_flag = 'val'
    test_valid_transforms = None
    target_transform = None
    val_dataset = DataGenerater(val_data_info_path, val_pre_fix_path, test_valid_transforms, val_flag, target_transform)

    return train_dataset, val_dataset

if __name__ == '__main__':

    # Here we use 8 fold cross validation, save_num means to use dataset0x as the validation set
    save_num = 1
    train_dataset, val_dataset = get_dataset(save_num=save_num)
    curr_model_name = "ostiapoints_net"
    model = OstiapointsNet()

    batch_size = 64
    num_workers = 16

    criterion = torch.nn.MSELoss()
    inital_lr = 0.001

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=inital_lr,weight_decay=0.001)

    trainer = Trainer(batch_size,
                      num_workers,
                      train_dataset,
                      val_dataset,
                      model,
                      curr_model_name,
                      optimizer,
                      criterion,
                      save_num=save_num,
                      start_epoch=0,
                      max_epoch=100,
                      initial_lr=inital_lr)

    trainer.run_train()