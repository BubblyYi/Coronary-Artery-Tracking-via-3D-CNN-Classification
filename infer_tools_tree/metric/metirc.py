# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 15:38
# @Author  : BubblyYi
# @FileName: metric.py
# @Software: PyCharm

import numpy as np
import copy
from metric.dijstra import correspondence
import time
import sys
def overlapToScore(overlap, observer):
    if overlap * 100 >= 99.95:
        return 100
    else:
        if overlap <= observer:
            score = (overlap / observer) * 50.0
            return score
        else:
            score = 50 + 50 * (overlap - observer) / (1.0 - observer)
            return score
def accuracyTOScore(accuracy, observer):
    if (accuracy < 1e-6) and (observer < 1e-6):
        return 100
    else:
        if accuracy <= observer:
            score = 100.0 - 50.0 * (accuracy / observer)
            return score
        else:
            score = (observer / accuracy) * 50.0
            return score


def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def pathLength(ref):
    if len(ref) < 2:
        return 0.0
    cumlen = 0
    for i in range(1, len(ref)):
        cumlen += get_distance(ref[i - 1], ref[i])
    return cumlen


def resampler(path, samplingDistance):
    result_path = np.array([]).reshape(0, 3)
    curLen = 0.0
    sampleNr = 0
    curIdx = 0
    while curIdx < len(path) - 1:
        nextLen = curLen + get_distance(path[curIdx + 1], path[curIdx])
        while sampleNr * samplingDistance < nextLen:
            dist = sampleNr * samplingDistance
            a = (nextLen - dist) / (nextLen - curLen)
            b = (dist - curLen) / (nextLen - curLen)
            c = copy.deepcopy(a * path[curIdx] + b * path[curIdx + 1]).reshape(1, 3)
            result_path = np.concatenate([result_path, c], axis=0)
            sampleNr += 1
        curLen = nextLen
        curIdx += 1
    result_path = np.concatenate([result_path, copy.deepcopy(path[-1]).reshape(1, 3)], axis=0)
    return result_path


def startCLFrom(s, n, rad, cl):
    startAt = -1
    i = 0
    j = 1
    while startAt < 0 and j != len(cl) - 1:
        if ((cl[i] - s) * n).sum() * ((cl[j] - s) * n).sum() < 0.0:
            a = ((cl[i] - s) * n).sum()
            b = (-1.0 * (cl[j] - s) * n).sum()
            pos = b / (a + b) * cl[i] + a / (a + b) * cl[j]
            if get_distance(pos, s) < 2.0 * rad:
                startAt = j
        i += 1
        j += 1
    return startAt if startAt >= 0 else 0


def calculate_metric(ds_nr, vs_nr, refer_path, infer_path, observer_path="observerscores.txt",
                     save_info_path="save_info.txt"):
    OV_ID = 0
    OF_ID = 1
    OT_ID = 2

    # Radius threshold for clinically relevant vessels (diameter >1.5mm is clinically relevant)
    radius_threshold = 1.5 / 2.0
    overlap_io_read = np.loadtxt(observer_path)
    overlap_io = np.ones((8, 4, 3)) * 2
    for i in overlap_io_read:
        ds = int(i[0])
        vs = int(i[1])
        ov = i[2]
        of = i[3]
        ot = i[4]
        overlap_io[ds][vs][0] = ov
        overlap_io[ds][vs][1] = of
        overlap_io[ds][vs][2] = ot
    OV_io = overlap_io[ds_nr][vs_nr][OV_ID]
    OF_io = overlap_io[ds_nr][vs_nr][OF_ID]
    OT_io = overlap_io[ds_nr][vs_nr][OT_ID]

    refer_all = np.loadtxt(refer_path)
    ref = refer_all[..., 0:3]
    rad = refer_all[..., 3]
    io = refer_all[..., 4]
    cl = np.loadtxt(infer_path)
    refSampling = pathLength(ref) / (len(ref) - 1)
    print("Reference sampling distance = {}".format(refSampling))
    clSampling = pathLength(cl) / (len(cl) - 1)
    print("Infer sampling distance = {}".format(clSampling))
    if refSampling / clSampling < 0.999 or refSampling / clSampling > 1.001:
        print("Resampling Infer ...")
        cl = resampler(cl, refSampling)

    clipUpTo = startCLFrom(ref[0], ref[1] - ref[0], rad[0], cl)
    cl = cl[clipUpTo:]
    t1 = time.time()
    print("connection start")
    connections = correspondence(ref=ref, infer=cl)
    t2 = time.time()
    print("connection time {}".format(t2 - t1))
    TPRs = np.zeros(len(ref), dtype=bool)
    TPMs = np.zeros(len(cl), dtype=bool)
    first_error_on_ref = len(ref)
    last_ref_in_large_vessel = -1
    last_cl_in_large_vessel = -1
    last_conn_in_large_vessel = -1
    whole_vessel_within_radius_threshold = False
    connectionLength = []
    for i_index, i in enumerate(connections):  # 发现有些地方匹配得有错误？
        length = get_distance(ref[i[0]], cl[i[1]])
        connectionLength.append(length)
        if length <= rad[i[0]]:
            TPRs[i[0]] = True
            TPMs[i[1]] = True
        else:
            if i[0] < first_error_on_ref:
                first_error_on_ref = i[0]
        if not whole_vessel_within_radius_threshold and rad[i[0]] >= radius_threshold:
            last_ref_in_large_vessel = i[0]
            last_cl_in_large_vessel = i[1]
            last_conn_in_large_vessel = i_index
            if i[0] == TPRs.size - 1:
                whole_vessel_within_radius_threshold = True
    TPR = TPRs.sum()
    TPM = TPMs.sum()
    FN = TPRs.size - TPR
    FP = TPMs.size - TPM
    OV = (TPM + TPR) / (TPM + TPR + FN + FP)
    TPR_fe = first_error_on_ref
    FN_fe = TPRs.size - first_error_on_ref
    OF = TPR_fe / (TPR_fe + FN_fe)
    TPR_t = TPRs[:last_ref_in_large_vessel + 1].sum()
    TPM_t = TPMs[:last_cl_in_large_vessel + 1].sum()
    FN_t = last_ref_in_large_vessel + 1 - TPR_t
    FP_t = last_cl_in_large_vessel + 1 - TPM_t
    OT = 1
    if (TPM_t + TPR_t + FN_t + FP_t) > 0:
        OT = (TPM_t + TPR_t) / (TPM_t + TPR_t + FN_t + FP_t)
    ADcount = 0
    ADsum = 0.0
    ADscoresum = 0.0
    AIcount = 0
    AIsum = 0.0
    AIscoresum = 0.0
    ATcount = 0
    ATsum = 0.0
    ATscoresum = 0.0
    for i in range(len(connections)):
        refidx = connections[i][0]
        length = connectionLength[i]
        score_AD = accuracyTOScore(length, io[refidx])

        ADsum += length
        ADscoresum += score_AD
        ADcount += 1
        if length <= rad[refidx]:
            score_AI = accuracyTOScore(length, io[refidx])
            AIsum += length
            AIscoresum += score_AI
            AIcount += 1
        if i <= last_conn_in_large_vessel:
            score_AT = accuracyTOScore(length, io[refidx])
            ATsum += length
            ATscoresum += score_AT
            ATcount += 1
    AD = 0.0
    ADscore = 0
    if ADcount > 0:
        AD = ADsum / ADcount
        ADscore = ADscoresum / ADcount
    # AveragE distance and score inside the vessel
    AI = 0.0
    AIscore = 0
    if AIcount > 0:
        AI = AIsum / AIcount
        AIscore = AIscoresum / AIcount
    #  Average distance and score in the section assumed to be clinically relevant
    AT = 0.0
    ATscore = 100
    if ATcount > 0:
        AT = ATsum / ATcount
        ATscore = ATscoresum / ATcount
    print("Dataset: {}".format(ds_nr))
    print("Vessel: {}".format(vs_nr))
    print("Accuracy AD: {:.6f},score: {:.4f}".format(AD, ADscore))
    print("Accuracy AI: {:.6f},score: {:.4f}".format(AI, AIscore))
    print("Accuracy AT: {:.6f},score: {:.4f}".format(AT, ATscore))
    print("Overlap OV: {:.6f}, score: {:.4f}".format(OV, overlapToScore(OV, OV_io)))
    print("Overlap OF: {:.6f}, score: {:.4f}".format(OF, overlapToScore(OF, OF_io)))
    print("Overlap OT: {:.6f}, score: {:.4f}".format(OT, overlapToScore(OT, OT_io)))
    stand_out = sys.stdout
    sys.stdout = open(save_info_path, "w")
    print("Dataset: {}".format(ds_nr))
    print("Vessel: {}".format(vs_nr))
    print("Accuracy AD: {:.6f},score: {:.4f}".format(AD, ADscore))
    print("Accuracy AI: {:.6f},score: {:.4f}".format(AI, AIscore))
    print("Accuracy AT: {:.6f},score: {:.4f}".format(AT, ATscore))
    print("Overlap OV: {:.6f}, score: {:.4f}".format(OV, overlapToScore(OV, OV_io)))
    print("Overlap OF: {:.6f}, score: {:.4f}".format(OF, overlapToScore(OF, OF_io)))
    print("Overlap OT: {:.6f}, score: {:.4f}".format(OT, overlapToScore(OT, OT_io)))
    sys.stdout = stand_out


if __name__ == '__main__':
    calculate_metric(1, 0, refer_path="reference.txt", infer_path="infer.txt")
