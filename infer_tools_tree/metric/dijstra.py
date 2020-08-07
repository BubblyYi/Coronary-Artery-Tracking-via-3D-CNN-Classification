# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 15:38
# @Author  : BubblyYi
# @FileName: dijstra.py
# @Software: PyCharm

import heapq
import numpy as np
import copy


def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class QueueElement:
    def __init__(self, dis, connection: list):
        self.dis = dis
        self.connection = connection

    def __lt__(self, other):
        return self.dis < other.dis

def correspondence(ref, infer):
    """

    :param ref: 参考点
    :param infer:
    :return:
    """
    refer_size = len(ref)
    infer_size = len(infer)

    if infer == [] or ref == []:
        print("data error")

    Done = 0
    OnFront = 1
    NotVisited = 2
    nodeStatus = np.ones((refer_size, infer_size)) * NotVisited  # 记录点访问状态的数组
    distanceMap = np.ones((refer_size, infer_size)) * float("inf")  # 距离数组
    prevPointer = np.zeros((refer_size, infer_size))  # 用以对匹配点进行回溯

    #
    q = []
    dist = get_distance(ref[0], infer[0])
    priorityQueue = QueueElement(dist, [0, 0])
    heapq.heappush(q, priorityQueue)  # 建立优先级队列
    nodeStatus[0][0] = OnFront
    distanceMap[0][0] = dist
    while q and q[0].dis < distanceMap[-1][-1]:
        queueElem = copy.deepcopy(q[0].connection)
        dist = q[0].dis
        heapq.heappop(q)
        while q and nodeStatus[queueElem[0]][queueElem[1]] == Done:  # 优先级队列不为空而且该点已经被访问过
            queueElem = copy.deepcopy(q[0].connection)
            dist = q[0].dis
            heapq.heappop(q)

        if nodeStatus[queueElem[0]][queueElem[1]] == Done:
            break
        if dist > distanceMap[-1][-1]:
            break
        nodeStatus[queueElem[0]][queueElem[1]] = Done
        distanceMap[queueElem[0]][queueElem[1]] = dist
        if queueElem[1] < infer_size - 1:
            newDist = dist + get_distance(ref[queueElem[0]], infer[queueElem[1] + 1])
            if nodeStatus[queueElem[0]][queueElem[1] + 1] == Done:
                continue
            elif nodeStatus[queueElem[0]][queueElem[1] + 1] == OnFront:
                if newDist >= distanceMap[queueElem[0]][queueElem[1] + 1]:
                    continue
            nodeStatus[queueElem[0]][queueElem[1] + 1] = OnFront
            distanceMap[queueElem[0]][queueElem[1] + 1] = newDist
            prevPointer[queueElem[0]][queueElem[1] + 1] = 2
            heapq.heappush(q, QueueElement(newDist, [queueElem[0], queueElem[1] + 1]))
        if queueElem[0] < refer_size - 1:
            newDist = dist + get_distance(ref[queueElem[0] + 1], infer[queueElem[1]])
            if nodeStatus[queueElem[0] + 1][queueElem[1]] == Done:
                continue
            elif nodeStatus[queueElem[0] + 1][queueElem[1]] == OnFront:
                if newDist >= distanceMap[queueElem[0] + 1][queueElem[1]]:
                    continue
            nodeStatus[queueElem[0] + 1][queueElem[1]] = OnFront
            distanceMap[queueElem[0] + 1][queueElem[1]] = newDist
            prevPointer[queueElem[0] + 1][queueElem[1]] = 1
            heapq.heappush(q, QueueElement(newDist, [queueElem[0] + 1, queueElem[1]]))

    revPath = []
    revPath.append([refer_size - 1, infer_size - 1])
    while revPath[-1][0] or revPath[-1][1]:
        pointer = prevPointer[revPath[-1][0]][revPath[-1][1]]
        if pointer == 1:
            revPath.append([revPath[-1][0] - 1, revPath[-1][1]])
        elif pointer == 2:
            revPath.append([revPath[-1][0], revPath[-1][1] - 1])
        else:
            raise ValueError
    revPath.reverse()
    return revPath
