import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils



imgs_id_list,labels_list,tarlabel_list=utils.load_ground_truth('dataset/dev_dataset.csv')
img_num=len(imgs_id_list)

lamda=1.0



eps=16.0/255.0

#获取插入特征图像
concat_img = cv2.imread('img/concat_img.png')[..., ::-1].astype(np.float32)
concat_img_guass = (concat_img - cv2.GaussianBlur(concat_img, (17, 17), 4))/255.0
#对每一张图像进行插入
for i in range(img_num):
    img = cv2.imread('img.png')[..., ::-1].astype(np.float32)
    img = cv2.resize(img, (299, 299))
    orgin = img.copy() / 255.0
    img_guass = cv2.GaussianBlur(img, (17, 17), 4) / 255.0

    adv = img_guass + lamda * concat_img_guass
    adv = orgin + np.clip(adv - orgin, -eps, eps)

    adv = (adv * 255).astype(np.int)
    plt.imshow(adv)
    plt.show()
    adv = adv[:, :, (2, 1, 0)]

    cv2.imwrite('adv.png', adv)











