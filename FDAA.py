from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io

imgs_id_list,labels_list,tarlabel_list=utils.load_ground_truth('dataset/dev_dataset.csv')
img_num=len(imgs_id_list)
#对每一张图像进行插入
for i in range(img_num):
    saliency=cv2.resize(cv2.imread('./dataset/Saliency_region/'+imgs_id_list[i]+'.png')[..., ::-1].astype(np.float32),(299,299))
    NAA = cv2.resize(cv2.imread('./adv/NAAs/inception_v3/'+imgs_id_list[i]+'.png')[..., ::-1].astype(np.float32), (299, 299))
    HIT = cv2.resize(cv2.imread('./adv/HIT/HIT6/' + imgs_id_list[i] + '.png')[..., ::-1].astype(np.float32), (299, 299))

    black = np.zeros_like(HIT)
    #根据显著图，框出显著区域的位置
    _,(top, bottom, left, right)=utils.bounding_box(NAA,saliency,80,1)
    #还原显著区域的原始大小
    resize=cv2.resize(NAA,(right-left+1,bottom-top+1))
    #将显著区域填充回原来的区域
    black[top:bottom+1,left:right+1,:]=resize

    saliency_map=np.where(saliency>5,1,0)
    #背景区域选择HIT
    adv=black*saliency_map+HIT*np.abs(1-saliency_map)

    # plt.subplot(221)
    # plt.axis('off')
    # plt.imshow(saliency)
    # plt.subplot(222)
    # plt.axis('off')
    # plt.imshow(black/255)
    # plt.subplot(223)
    # plt.axis('off')
    # plt.imshow(adv/255)
    # plt.subplot(224)
    # plt.axis('off')
    # plt.imshow(NAA/255)
    # plt.show()
    adv = adv[:, :, (2, 1, 0)]
    cv2.imwrite('./adv/region_attack/inception_v3/' + imgs_id_list[i] + '.png',adv)



