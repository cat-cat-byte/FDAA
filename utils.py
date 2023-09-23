# coding: utf-8
""""
utils file
用于对抗样本攻击过程中常用的函数：
1.正则化函数：vgg_normalization,inception_normalization
作用：在图像送入神经网络之前，需要对图像进行正则化。
2.逆正则化函数：inv_vgg_normalization,inv_inception_normalization
作用：将正则化的图像，映射回正常图像
3.（逆）正则化映射函数：normalization_fn_map,inv_inception_normalization
作用：利用网络名称，直接映射到对应的正则化函数。
4.标签偏移映射函数：offset
作用：有点网络的label从零起始，有的则从1起始。标签偏移函数根据输入的网络名称，自动选择对应的偏移量。
5.图像size映射函数：image_size
作用：网络输入的默认尺寸
6.权重存储路径映射函数：checkpoint_paths
作用：根据网络名称，映射对应的存储路径。
7.图像加载函数：load_image(image_path, image_size, batch_size),load_image_csv(image_path, image_size, batch_size,csv_path)
作用：按照图像大小与batch_size加载图像
8.图像存储函数：save_image(images,filename,output_dir)

"""
import csv
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, resnet_v1, vgg, nets_factory
slim = tf.contrib.slim
def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]
def densenet_normalization(image):
    image[:,:, :, 0] = (image[:,:, :, 0] - 123.68)
    image[:,:, :, 1] = (image[:,:, :, 1] - 116.78)
    image[:,:, :, 2] = (image[:,:, :, 2] - 103.94)
    return image*0.017
def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2.
def inv_vgg_normalization(image):
  return np.clip(image + [123.68, 116.78, 103.94],0,255)
def inv_densenet_normalization(image):
    return np.clip(image/0.017 + [123.68, 116.78, 103.94],0,255)
def inv_inception_normalization(image):
  return np.clip((image + 1.0) * 0.5 * 255,0,255)

normalization_fn_map = {
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': inception_normalization,
    'resnet_v2_101': inception_normalization,
    'resnet_v2_152': inception_normalization,
    'resnet_v2_200': inception_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
    'densenet121':inception_normalization,
    'densenet161':inception_normalization,
    'densenet169':inception_normalization,
    'X101-DA':inception_normalization,
    'R152-B':inception_normalization,
    'R152-D':inception_normalization,
}

inv_normalization_fn_map = {
    'inception_v1': inv_inception_normalization,
    'inception_v2': inv_inception_normalization,
    'inception_v3': inv_inception_normalization,
    'inception_v4': inv_inception_normalization,
    'inception_resnet_v2': inv_inception_normalization,
    'resnet_v1_50': inv_vgg_normalization,
    'resnet_v1_101': inv_vgg_normalization,
    'resnet_v1_152': inv_vgg_normalization,
    'resnet_v1_200': inv_vgg_normalization,
    'resnet_v2_50': inv_inception_normalization,
    'resnet_v2_101': inv_inception_normalization,
    'resnet_v2_152': inv_inception_normalization,
    'resnet_v2_200': inv_inception_normalization,
    'vgg_16': inv_vgg_normalization,
    'vgg_19': inv_vgg_normalization,
    'densenet121': inv_densenet_normalization,
    'densenet161': inv_densenet_normalization,
    'densenet169': inv_densenet_normalization,
    'X101-DA':inv_densenet_normalization,
    'R152-B':inv_densenet_normalization,
    'R152-D':inv_densenet_normalization,
}

offset = {
    'inception_v1': 1,
    'inception_v2': 1,
    'inception_v3': 1,
    'inception_v4': 1,
    'inception_resnet_v2': 1,
    'resnet_v1_50': 0,
    'resnet_v1_101': 0,
    'resnet_v1_152': 0,
    'resnet_v1_200': 0,
    'resnet_v2_50': 1,
    'resnet_v2_101': 1,
    'resnet_v2_152': 1,
    'resnet_v2_200': 1,
    'vgg_16': 0,
    'vgg_19': 0,
    'densenet121': 0,
    'densenet161': 0,
    'densenet169': 0,
    'X101-DA':1,
    'R152-B': 1,
    'R152-D': 1,
  }

image_size={
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'inception_v4': 299,
    'inception_resnet_v2': 299,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg_16': 224,
    'vgg_19': 224,
    'densenet121': 299,
    'densenet161': 299,
    'densenet169': 299,
    'X101-DA':299,
    'R152-B': 299,
    'R152-D': 299,
  }
base_path='./models_tf'

checkpoint_paths = {
    'inception_v1': None,
    'inception_v2': None,
    'inception_v3': base_path+'/inception_v3.ckpt',
    'inception_v4': base_path+'/inception_v4.ckpt',
    'inception_resnet_v2': base_path+'/inception_resnet_v2_2016_08_30.ckpt',
    'resnet_v1_50': base_path+'/resnet_v1_50.ckpt',
    'resnet_v1_101': None,
    'resnet_v1_152': base_path+'/resnet_v1_152.ckpt',
    'resnet_v1_200': None,
    'resnet_v2_50': base_path+'/resnet_v2_50.ckpt',
    'resnet_v2_101': base_path+'/resnet_v2_101.ckpt',
    'resnet_v2_152': base_path+'/resnet_v2_152.ckpt',
    'resnet_v2_200': None,
    'vgg_16': base_path+'/vgg_16.ckpt',
    'vgg_19': base_path+'/vgg_19.ckpt',
    'densenet121': base_path+'/tf-densenet121.ckpt',
    'densenet161': base_path+'/tf-densenet161.ckpt',
    'densenet169': base_path+'/tf-densenet169.ckpt',
    'adv_inception_v3':base_path+'/adv_inception_v3.ckpt',
    'adv_inception_resnet_v2':base_path+'/adv_inception_resnet_v2.ckpt',
    'ens3_adv_inception_v3':base_path+'/ens3_adv_inception_v3.ckpt',
    'ens4_adv_inception_v3':base_path+'/ens4_adv_inception_v3.ckpt',
    'ens_adv_inception_resnet_v2':base_path+'/ens_adv_inception_resnet_v2.ckpt',
    'pnasnet':base_path+'/pnasnet.ckpt',
    'X101-DA': base_path+'X101-DenoiseAll_rename.npz',
    'R152-B': base_path+'R152_rename.npz',
    'R152-D': base_path+'R152_Denoise.npz',
  }

def load_image(image_path, image_size, batch_size):
    images = []
    filenames=[]
    labels=[]
    idx=0

    files=os.listdir(image_path)
    files.sort(key=lambda x: int(x[:-4]))
    for i,filename in enumerate(files):
        # image = imread(image_path + filename)
        # image = imresize(image, (image_size, image_size)).astype(np.float)
        image=Image.open(image_path + filename)
        image=image.resize((image_size,image_size))
        image=np.array(image)
        images.append(image)
        filenames.append(filename)

        labels.append(int(ground_truth[i]))
        idx+=1
        if idx==batch_size:
            yield np.array(images),np.array(filenames),np.array(labels)
            idx=0
            images=[]
            filenames=[]
            labels=[]
    if idx>0:
        yield np.array(images), np.array(filenames),np.array(labels)

def save_image(images,filenames,output_dir):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(filenames):
        # imsave(output_dir+name,images[i].astype('uint8'))
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list

def load_image_csv(image_path, image_size, batch_size,csv_path):
    image_id,label_id,target_id=load_ground_truth(csv_path)
    images = []
    filenames = []
    labels = []
    idx = 0
    for i in range(len(image_id)):
        filename = image_id[i] + '.png'
        image = Image.open(image_path + filename)
        image = image.resize((image_size, image_size))
        image = np.array(image)

        images.append(image)
        filenames.append(filename)
        labels.append(int(label_id[i]))
        idx += 1
        if idx == batch_size:
            yield np.array(images), np.array(filenames), np.array(labels)
            idx = 0
            images = []
            filenames = []
            labels = []
    if idx > 0:
        yield np.array(images), np.array(filenames), np.array(labels)

def bounding_box(img,cam,threshold,type):
    """type=1,bounding_img返回显著区域；type=2，bounding_img返回显著区域的图像"""
    left,right,top,bottom = 0,0,0,0
    height,width,_=img.shape
    for i in range(width):
        if any((cam[:,i,0]>threshold)):
            left = i
            break
    for i in range(width):
        j=width-i-1
        if any(cam[:,j,0]>threshold):
            right=j
            break
    for i in range(height):
        if any(cam[i,:,0]>threshold):
            top=i
            break
    for i in range(height):
        j=height-i-1
        if any(cam[j,:,0]>threshold):
            bottom=j
            break
    if type==1:
        bounding_img = np.zeros([height, width])
        bounding_img[top:bottom, left:right] = 255
    elif type==2:
        bounding_img = img[top:bottom, left:right, :]
    return bounding_img,(top,bottom,left,right)


def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad

if __name__=='__main__':
    pass


