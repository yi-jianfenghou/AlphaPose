#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:03:10 2018

@author: spl
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from PIL import Image
import os


#return box_size
def box_size(one_json,num):
    height = int(one_json[num][2] - one_json[num][0])
    if (height % 2) != 0:
        height += 1
    width = int(one_json[num][3] - one_json[num][1])
    if (width % 2) != 0:
        width += 1
    return height,width

#return people hm
def hmbox_People(one_hms,num):
    hmpeople = one_hms[num][0]
    for i in range(16):
        hmpeople += one_hms[num][i+1]
    return hmpeople

#integration hm and image
def integration_box(imagepath,jsonpath,npypath):
    with open(jsonpath, 'r') as file:
        one_json = json.load(file)
    one_hms = np.load(npypath)
    tfimage = cv2.imread(imagepath)
    heightborde = int(len(tfimage)/2)
    widthborde = int(len(tfimage[0])/2)
    tfimage = cv2.copyMakeBorder(tfimage,heightborde, heightborde, widthborde, widthborde ,cv2.BORDER_CONSTANT,value=(0,0,0))
    for num in range(len(one_json)):
        #get heatmap box and width&height
        height,width = box_size(one_json,num)
        hmpeople = hmbox_People(one_hms,num)
        #hm numpy to rgb then resize
        hmpeople *= 225
        hmpeople = Image.fromarray(hmpeople.astype('uint8')).convert('RGB')
        cvhmPeople = cv2.cvtColor(np.array(hmpeople), cv2.COLOR_RGB2BGR)
        cvhmPeople = cv2.resize(cvhmPeople,(width,height))
        #process tf image
        npcrop = tfimage[int(one_json[num][1])+heightborde:int(one_json[num][1])+height+heightborde,int(one_json[num][0])+widthborde:int(one_json[num][0])+width+widthborde]
        AddImg = cvhmPeople+npcrop
        tfimage[int(one_json[num][1])+heightborde:int(one_json[num][1])+height+heightborde,int(one_json[num][0])+widthborde:int(one_json[num][0])+width+widthborde] = AddImg
    tfimage = tfimage[heightborde:heightborde*3,widthborde:widthborde*3]
    tfimage = Image.fromarray(cv2.cvtColor(tfimage,cv2.COLOR_BGR2RGB))
    return tfimage

if __name__ == "__main__":
    #load data
    file_dirs = "/home/spl/ruiyang/Alphapose/out_npy/"
    image_dirs = "/home/spl/ruiyang/Alphapose/examples/demo/"
    file_list = []
    image_list = []
    for file in os.listdir(file_dirs):
        file_list.append(file)  # 1jpg.json 1jpg.npy ...
    for file in os.listdir(image_dirs):
        image_list.append(file)  # 1.jpg, 2.jpg ...
    file_list.sort()
    image_list.sort()
   # for i in range(len(image_list)):
    for i in range(3):
        imagepath = image_dirs + image_list[i]
        jsonpath = file_dirs + file_list[2*i]
        npypath = file_dirs + file_list[2*i+1]
        ResImg = integration_box(imagepath,jsonpath,npypath)
        plt.imshow(ResImg)
      #  print(image_dirs + image_list[i])
      #  print(file_dirs + file_list[2*i])
      #  print(file_dirs + file_list[2*i+1])
      #  print('-------')
