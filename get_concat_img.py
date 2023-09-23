import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2

#生成复合版本的HIT图像
def get_concat(orgin,x,y):
    width = orgin.shape[0]//y    #图像的高
    height = orgin.shape[1]//x   #图像的宽
    orgin=cv2.resize(orgin,(width,height),interpolation=cv2.INTER_AREA)
    concat=np.zeros([x*orgin.shape[0],y*orgin.shape[1],orgin.shape[2]])
    for i in range(x):
        for j in range(y):
            concat[i*width:(i+1)*width,j*height:(j+1)*height,:]=orgin
    return concat

#读取基础的HIT图像
img= cv2.imread('img/pic.png')[..., ::-1]
img=cv2.resize(img,(300,300),interpolation=cv2.INTER_AREA)
concat_img=cv2.resize(get_concat(img,1,1).astype("uint8"),(299,299),interpolation=cv2.INTER_AREA)
r,g,b = cv2.split(concat_img)
concat_img=cv2.merge([b,g,r])
cv2.imwrite('img/concat_img1.png', concat_img)

