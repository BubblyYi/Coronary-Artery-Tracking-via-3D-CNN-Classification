# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 17:43
# @Author  : BubblyYi、QYD
# @FileName: build_vessel_tree.py
# @Software: PyCharm
import numpy as np
import torch
import copy
from setting import infer_model, device, spacing, re_spacing_img, resize_factor, max_points, prob_thr
from utils import get_spacing_res2, data_preprocess, prob_terminates, get_shell, get_angle
from tree import TreeNode

def search_tree(root: TreeNode, point):
    '''
    BFS tree, determine whether the current input point is close to the existing centerline
    :param root:
    :param point:
    :return:
    '''
    queue = []
    queue.append(root)
    while queue:
        vertex = queue.pop(0)
        point = np.array(point)
        dis_all = np.linalg.norm(point - vertex.value, axis=1)
        dis = dis_all.min()
        index = dis_all.argmin()
        if dis < 3:
            return vertex, index
        nodes = vertex.child_list
        for w in nodes:
            queue.append(w)
    return None


def dfs_search_tree(root: TreeNode):
    '''
    DFS, build single vessel
    :param root:
    :return: single vessel
    '''
    stack_list = []
    visited = []
    stack_list.append(root)
    visited.append(root)
    res = [root]
    single_vessel = []
    while len(stack_list) > 0:
        temp = []
        x = stack_list[-1]
        for w in x.child_list:
            if w not in visited:
                temp.append(w)
                visited.append(w)
                stack_list.append(w)
                break
        if len(temp) > 0:
            res.append(temp[0])
        if stack_list[-1] == x:
            single_vessel.append(res[:])
            res.pop()
            stack_list.pop()
    return single_vessel


def infer(start: list):
    """
    :param start: Initial point
    :return: Moving position, the index of maximum confidence direction, Current termination probability
    """
    max_z = re_spacing_img.shape[0]
    max_x = re_spacing_img.shape[1]
    max_y = re_spacing_img.shape[2]

    cut_size = 9
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    center_x_pixel = get_spacing_res2(start[0], spacing_x, resize_factor[1])
    center_y_pixel = get_spacing_res2(start[1], spacing_y, resize_factor[2])
    center_z_pixel = get_spacing_res2(start[2], spacing_z, resize_factor[0])

    left_x = center_x_pixel - cut_size
    right_x = center_x_pixel + cut_size
    left_y = center_y_pixel - cut_size
    right_y = center_y_pixel + cut_size
    left_z = center_z_pixel - cut_size
    right_z = center_z_pixel + cut_size

    new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))

    if not (
            left_x < 0 or right_x < 0 or left_y < 0 or right_y < 0 or left_z < 0 or right_z < 0 or left_x >= max_x or right_x >= max_x or left_y >= max_y or right_y >= max_y or left_z >= max_z or right_z >= max_z):
        for ind in range(left_z, right_z + 1):
            src_temp = re_spacing_img[ind].copy()
            new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
        input_data = data_preprocess(new_patch)

        inputs = input_data.to(device)
        outputs = infer_model(inputs.float())

        outputs = outputs.view((len(input_data), max_points + 1))
        outputs_1 = outputs[:, :len(outputs[0]) - 1]
        outputs_2 = outputs[:, -1]

        outputs_1 = torch.nn.functional.softmax(outputs_1, 1)
        indexs = np.argsort(outputs_1.cpu().detach().numpy()[0])[::-1]
        curr_prob = prob_terminates(outputs_1, max_points).cpu().detach().numpy()[0]
        curr_r = outputs_2.cpu().detach().numpy()[0]
        sx, sy, sz = get_shell(max_points, curr_r)
        return [sx, sy, sz], indexs, curr_r, curr_prob
    else:
        return None


def search_first_node(start: list, prob_records: list):
    """
    :param start: Initial point
    :return: Next direction vector, Probability record, Current radius
    """
    s_all, indexs, curr_r, curr_prob = infer(start=start)
    start_x, start_y, start_z = start
    prob_records.pop(0)
    prob_records.append(curr_prob)
    sx, sy, sz = s_all
    forward_x = sx[indexs[0]] + start_x
    forward_y = sy[indexs[0]] + start_y
    forward_z = sz[indexs[0]] + start_z
    forward_move_direction_x = sx[indexs[0]]
    forward_move_direction_y = sy[indexs[0]]
    forward_move_direction_z = sz[indexs[0]]
    for i in range(1, len(indexs)):
        curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                               np.array([forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]))
        # To determine two initial opposing directions of the tracker, two local maxima d0 and d0′ separated by an angle ≥ 90°
        if curr_angle >= 90:
            backward_move_direction_x = copy.deepcopy(sx[indexs[i]])
            backward_move_direction_y = copy.deepcopy(sy[indexs[i]])
            backward_move_direction_z = copy.deepcopy(sz[indexs[i]])
            break
    backward_x = backward_move_direction_x + start_x
    backward_y = backward_move_direction_y + start_y
    backward_z = backward_move_direction_z + start_z
    direction = {}
    direction["forward"] = [forward_x, forward_y, forward_z]
    direction["forward_vector"] = [forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]
    direction["backward"] = [backward_x, backward_y, backward_z]
    direction["backward_vector"] = [backward_move_direction_x, backward_move_direction_y, backward_move_direction_z]
    return direction, prob_records, curr_r


def move(start: list, shell_arr: list, indexs: list, move_direction: list):
    """
    Moving ball
    :param start: start point
    :param shell_arr: shell arr
    :param indexs: index of next direction
    :param move_direction: last move direction
    :param curr_r: radius
    :return: direction vector, move to next point
    """
    start_x, start_y, start_z = start
    sx, sy, sz = shell_arr
    move_direction_x, move_direction_y, move_direction_z = move_direction
    for i in range(len(indexs)):
        curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                               np.array([move_direction_x, move_direction_y, move_direction_z]))
        # Only directions with an angle ≤ 60°to the previously followed direction are considered.
        if curr_angle <= 60:
            new_x = sx[indexs[i]] + start_x
            new_y = sy[indexs[i]] + start_y
            new_z = sz[indexs[i]] + start_z
            move_direction_x = sx[indexs[i]]
            move_direction_y = sy[indexs[i]]
            move_direction_z = sz[indexs[i]]
            break

    return [move_direction_x, move_direction_y, move_direction_z], [new_x, new_y, new_z]

def search_one_direction(start: list, move_direction: list, prob_records: list, point_list: list, r_list: list,
                         root: TreeNode, find_node=None):
    """
    :param start: start point
    :param move_direction: 上一次的移动方向 last move direction
    :param prob_records: record of termination probability
    :param point_list:
    :param r_list: radius arr
    :return:
    """
    find_node_initial = None
    prob_mean = sum(prob_records) / len(prob_records)
    while prob_mean <= prob_thr and find_node_initial is None:
        result = infer(start=start)
        if result is not None:
            shell_arr, indexs, curr_r, curr_prob = result
            r_list.append(curr_r)
            point_list.append(start)
            prob_records.pop(0)
            prob_records.append(curr_prob)
            prob_mean = sum(prob_records) / len(prob_records)
            move_direction, next_point = move(start=start, shell_arr=shell_arr, indexs=indexs,
                                         move_direction=move_direction)
            if find_node is None:
                find_node_initial = search_tree(root, next_point)
        else:
            break
    return find_node_initial


def search_line(start, curr_r, direction, prob_records, root: TreeNode):
    '''
    Search from the initial point to the direction of d0 and d0',
    :param start:
    :param curr_r:
    :param direction:
    :param prob_records:
    :param root:
    :return:
    '''
    point_list = []
    r_list = []
    point_list.append(start)
    r_list.append(curr_r)
    point_forward_list = copy.deepcopy(point_list)
    r_forward_list = copy.deepcopy(r_list)
    prob_forward_records = copy.deepcopy(prob_records)
    point_backward_list = copy.deepcopy(point_list)
    r_backward_list = copy.deepcopy(r_list)
    prob_backward_records = copy.deepcopy(prob_records)
    find_node_forward = search_one_direction(start=direction["forward"],
                                             move_direction=direction["forward_vector"],
                                             prob_records=prob_forward_records,
                                             r_list=r_forward_list, point_list=point_forward_list, root=root)
    find_node_backward = search_one_direction(start=direction["backward"],
                                              move_direction=direction["backward_vector"],
                                              prob_records=prob_backward_records, r_list=r_backward_list,
                                              point_list=point_backward_list, find_node=find_node_forward, root=root)
    find = True

    # If the current point is within 200 points from the end of the centerline,
    # it will be spliced with the current centerline, otherwise it will be set as a new branch
    add_thr = 200
    if find_node_forward is not None:
        point_forward_list.reverse()
        r_forward_list.reverse()
        point_list = point_forward_list + point_backward_list
        r_list = r_forward_list + r_backward_list
        res_arr = interpolation(point_list, r_list)
        start_point_index = find_node_forward[1]
        start_node = find_node_forward[0]
        start_coord = start_node.value[start_point_index]
        tmp_arr = np.linspace(start_coord, res_arr[0], num=100)
        res_arr = np.vstack((tmp_arr, res_arr))
        if start_node != root and start_point_index > start_node.value.shape[0] - add_thr:
            start_node.value = np.vstack((start_node.value[:start_point_index], res_arr))
        else:
            start_node.add_child(TreeNode(res_arr, start_point_index=start_point_index))
    elif find_node_backward is not None:
        point_backward_list.reverse()
        r_backward_list.reverse()
        point_list = point_backward_list + point_forward_list
        r_list = r_backward_list + r_forward_list
        res_arr = interpolation(point_list, r_list)
        start_point_index = find_node_backward[1]
        start_node = find_node_backward[0]
        start_coord = start_node.value[start_point_index]
        tmp_arr = np.linspace(start_coord, res_arr[0], num=100)
        res_arr = np.vstack((tmp_arr, res_arr))
        if start_node != root and start_point_index > start_node.value.shape[0] - add_thr:
            start_node.value = np.vstack((start_node.value[:start_point_index], res_arr))
        else:
            start_node.add_child(TreeNode(res_arr, start_point_index=start_point_index))
    else:
        # This vessel is added to the final record only when the ostia point is found
        find = False
    return find


def interpolation(point_list: list, r_list: list):
    # Interpolate according to 0.03 mm
    p1 = point_list[0]
    p2 = point_list[1]
    res_arr = np.linspace(p1, p2, num=int(r_list[0] / 0.03))
    for i in range(1, len(point_list) - 1):
        p1 = point_list[i]
        p2 = point_list[i + 1]
        tmp_arr = np.linspace(p1, p2, num=int(r_list[i] / 0.03))
        res_arr = np.vstack((res_arr, tmp_arr))
    return res_arr

def build_vessel_tree(seeds: np.ndarray, root: TreeNode):
    '''
    :param seeds:seeds arr
    :param root: tree root
    :return:
    '''
    prob_records = [0] * 3
    seeds_unused = []
    for seed in seeds:
        if search_tree(root, seed) is None:
            direction, prob_records, curr_r = search_first_node(start=seed, prob_records=prob_records)
            find = search_line(start=seed, curr_r=curr_r, prob_records=prob_records, direction=direction, root=root)
            if not find:
                seeds_unused.append(seed)
    for seed in seeds_unused:
        if search_tree(root, seed) is None:
            direction, prob_records, curr_r = search_first_node(start=seed, prob_records=prob_records)
            search_line(start=seed, curr_r=curr_r, prob_records=prob_records, direction=direction, root=root)
