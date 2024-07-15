import csv
import os
import cv2
import torch
from pathlib import Path, PurePosixPath
from datetime import datetime

from config_parser import ConfigParser

import sys
import numpy as np

import time

def IOU(bbox, gt):

    bbox_x2 = bbox[0] + bbox[2]
    bbox_y2 = bbox[1] + bbox[3]

    gt_x2 = gt[0] + gt[2]
    gt_y2 = gt[1] + gt[3]

    intersection_x1 = max(bbox[0], gt[0])
    intersection_y1 = max(bbox[1], gt[1])
    intersection_x2 = min(bbox_x2, gt_x2)
    intersection_y2 = min(bbox_y2, gt_y2)

    intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    box1_area = abs((bbox_x2 - bbox[0]) * (bbox_y2 - bbox[1]))
    box2_area = abs((gt_x2 - gt[0]) * (gt_y2 - gt[1]))

    return intersection / (box1_area + box2_area - intersection + 1e-7)
    # min_x1a = int(bbox[i][0])
    # min_y1a = int(bbox[i][1])
    # max_x1a = int(bbox[i][0]) + int(bbox[i][2])
    # max_y1a = int(bbox[i][1]) + int(bbox[i][3])
    #
    #
    # min_x2a = int(gt[i][0])
    # min_y2a = int(gt[i][1])
    # max_x2a = int(gt[i][0])+ int(gt[i][2])
    # max_y2a = int(gt[i][1]) + int(gt[i][3])
    #
    # reta = 0
    # # get area of rectangle A and B
    # rect1_areaa = (max_x1a - min_x1a) * (max_y1a - min_y1a)
    # rect2_areaa = (max_x2a - min_x2a) * (max_y2a - min_y2a)
    #
    # # get length and width of intersection
    # intersection_x_lengtha = min(max_x1a, max_x2a) - max(min_x1a, min_x2a)
    # intersection_y_lengtha = min(max_y1a, max_y2a) - max(min_y1a, min_y2a)
    #
    # # IoU
    # if intersection_x_lengtha > 0 and intersection_y_lengtha > 0:
    #     intersection_areaa = intersection_x_lengtha * intersection_y_lengtha
    #     union_areaa = rect1_areaa + rect2_areaa - intersection_areaa
    #     reta = intersection_areaa / union_areaa
    #
    # #rect.append(reta)
    # #print('reta',reta)
    #
    #
    #
    # min_x1b = int(bbox[i][4])
    # min_y1b = int(bbox[i][5])
    # max_x1b = int(bbox[i][4]) + int(bbox[i][6])
    # max_y1b = int(bbox[i][5]) + int(bbox[i][7])
    #
    # min_x2b = int(gt[i][4])
    # min_y2b = int(gt[i][5])
    # max_x2b = int(gt[i][4]) + int(gt[i][6])
    # max_y2b = int(gt[i][5]) + int(gt[i][7])
    #
    #
    # retb = 0
    # # get area of rectangle A and B
    # rect1_areab = (max_x1b - min_x1b) * (max_y1b - min_y1b)
    # rect2_areab = (max_x2b - min_x2b) * (max_y2b - min_y2b)
    #
    # # get length and width of intersection
    # intersection_x_lengthb = min(max_x1b, max_x2b) - max(min_x1b, min_x2b)
    # intersection_y_lengthb = min(max_y1b, max_y2b) - max(min_y1b, min_y2b)
    #
    # # IoU
    # if intersection_x_lengthb > 0 and intersection_y_lengthb > 0:
    #     intersection_areab = intersection_x_lengthb * intersection_y_lengthb
    #     union_areab = rect1_areab + rect2_areab - intersection_areab
    #     retb = intersection_areab / union_areab
    #
    #
    # # for k in range(3):
    # #     for i,j,m,n in range(3):
    # min_x1c =int(bbox[i][8])
    # min_y1c =int(bbox[i][9])
    # max_x1c =int(bbox[i][8])+int(bbox[i][10])
    # max_y1c = int(bbox[i][9]) + int(bbox[i][11])
    #
    # min_x2c = int(gt[i][8])
    # min_y2c = int(gt[i][9])
    # max_x2c = int(gt[i][8]) + int(gt[i][10])
    # max_y2c = int(gt[i][9]) + int(gt[i][11])
    #
    #
    #
    # retc = 0
    # # get area of rectangle A and B
    # rect1_areac = (max_x1c - min_x1c) * (max_y1c - min_y1c)
    # rect2_areac = (max_x2c - min_x2c) * (max_y2c - min_y2c)
    #
    # # get length and width of intersection
    # intersection_x_lengthc = min(max_x1c, max_x2c) - max(min_x1c, min_x2c)
    # intersection_y_lengthc = min(max_y1c, max_y2c) - max(min_y1c, min_y2c)
    #
    # # IoU
    # if intersection_x_lengthc > 0 and intersection_y_lengthc > 0:
    #     intersection_areac = intersection_x_lengthc * intersection_y_lengthc
    #     union_areac = rect1_areac + rect2_areac - intersection_areac
    #     retc = intersection_areac / union_areac
    #
    # rect.append(reta)
    # rect.append(retb)
    # rect.append(retc)
    # print(rect)
    #
    # return rect


def main():

    for i in range(202):
        gt = []
        coo = []

        f = open('E:\\cswintt_result\\newvideo1\\test\\twoperson_baseliine+multi_state\\bbox.csv', "r", encoding='utf-8-sig')
        reader = csv.reader(f)

        for gtcoor in reader:
            gt.append(gtcoor)

        f2 = open('D:\\gt\\65gt.csv', "r", encoding='utf-8-sig')
        reader = csv.reader(f2)
        next(reader)

        for coor in reader:
            # list(map(int))
            # arr = [int(coor[0])]
            # arr = str(arr)
            arr = [str(int(coor[0]))]
            coo.append(arr + coor[5:9])



        iou = IOU(coo,gt)

        result_iou = open('./ground_truth/twoperson_baseliine+multi_state_iou.csv', 'a', newline='')
        result_iou = csv.writer(result_iou)
        # result_iou.write(str(iou1) + ','+ str(iou2) +','+str(iou3))
        result_iou.writerow(iou)






if __name__ == "__main__":
    main()