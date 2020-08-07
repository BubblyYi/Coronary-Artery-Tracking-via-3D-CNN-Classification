# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 15:38
# @Author  : BubblyYi
# @FileName: tree.py
# @Software: PyCharm

class TreeNode(object):
    def __init__(self, value, start_point_index):
        self.value = value
        self.start_point_index = start_point_index
        self.child_list = []

    def add_child(self, node):
        self.child_list.append(node)