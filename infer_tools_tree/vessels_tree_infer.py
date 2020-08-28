# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 16:33
# @Author  : QYD
# @FileName: vessles_tree_infer.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generate_seeds_ositas import search_seeds_ostias
from setting import setting_info
from build_vessel_tree import build_vessel_tree, TreeNode, dfs_search_tree
from utils import save_info
import os

res_seeds, res_ostia = search_seeds_ostias()
seeds_gen_info_to_save = os.path.join(setting_info["seeds_gen_info_to_save"], "seeds.csv")
ostias_gen_info_to_save = os.path.join(setting_info["ostias_gen_info_to_save"], "ostias.csv")
infer_line_to_save = setting_info["infer_line_to_save"]
fig_to_save = setting_info["fig_to_save"]
reference_path = setting_info["reference_path"]
save_info(res_seeds, path=seeds_gen_info_to_save)
save_info(res_ostia, path=ostias_gen_info_to_save)
seeds = pd.read_csv(seeds_gen_info_to_save)[["x", "y", "z"]].values

ostias = []
head_node_list = pd.read_csv(ostias_gen_info_to_save)[["x", "y", "z"]].values
ostias_thr = 10
node_first = head_node_list[0]
ostias.append(node_first.tolist())
for node in head_node_list:
    if np.linalg.norm(node - node_first) > ostias_thr:
        ostias.append(node.tolist())
        break
if len(ostias)<2:
    print("not find 2 ostia points")
else:
    print("build vessel tree")
    root = TreeNode(ostias, start_point_index=None)
    build_vessel_tree(seeds, root=root)
    single_tree = dfs_search_tree(root)
    vessel_tree_postprocess = []
    for vessel_list in single_tree:
        vessel_list.pop(0)
        res = np.array([]).reshape(0, 3)
        while vessel_list:
            first_node = vessel_list[0]
            first_res = first_node.value
            vessel_list.pop(0)
            if vessel_list:
                second_node = vessel_list[0]
                first_res = first_res[:second_node.start_point_index]
                res = np.vstack((res, first_res))
            else:
                res = np.vstack((res, first_res))
                vessel_tree_postprocess.append(res)
    for i, vessel in enumerate(vessel_tree_postprocess):
        np.savetxt(infer_line_to_save + "/vessel_{}.txt".format(i), vessel)
    ax = plt.axes(projection='3d')
    for i, vessel in enumerate(vessel_tree_postprocess):
        ax.scatter(vessel[..., 0], vessel[..., 1], vessel[..., 2], label=" infer {}".format(i))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(fig_to_save + "/infer_tree_new.jpg")
    ax1 = plt.axes(projection='3d')
    for i in range(4):
        res = np.loadtxt(reference_path + "/vessel{}/reference.txt".format(i))
        ax1.scatter(res[..., 0], res[..., 1], res[..., 2], label="refer vessel {}".format(i))
    for i, vessel in enumerate(vessel_tree_postprocess):
        ax1.scatter(vessel[..., 0], vessel[..., 1], vessel[..., 2], label=" infer {}".format(i))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(fig_to_save + '/refer_infer_tree.jpg')
    ax2 = plt.axes(projection='3d')
    for i in range(4):
        res = np.loadtxt(reference_path + "/vessel{}/reference.txt".format(i))
        ax2.scatter(res[..., 0], res[..., 1], res[..., 2], label="refer vessel{}".format(i))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(fig_to_save + '/refer_tree.jpg')

    ax3 = plt.axes(projection='3d')
    seeds_point = pd.read_csv(seeds_gen_info_to_save)[["x", "y", "z"]].values
    ax3.scatter(seeds_point[..., 0], seeds_point[..., 1], seeds_point[..., 2], label="generation seeds_point")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(
        fig_to_save + '/seeds_points.jpg')

    ax4 = plt.axes(projection='3d')
    seeds_point = pd.read_csv(seeds_gen_info_to_save)[["x", "y", "z"]].values
    for i in range(4):
        res = np.loadtxt(reference_path + "/vessel{}/reference.txt".format(i))
        ax4.scatter(res[..., 0], res[..., 1], res[..., 2], label="refer vessel{}".format(i))
    ax4.scatter(seeds_point[..., 0], seeds_point[..., 1], seeds_point[..., 2], label="generation seeds_point")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(
        fig_to_save + "/seeds_points.jpg")
