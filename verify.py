from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
import imgaug.augmenters as aug
from attack_method import *
model_names=['inception_v3',	'inception_v4',	'inception_resnet_v2',	'resnet_v2_152',	'vgg_19',	'vgg_16',	'adv_inception_v3',	'adv_inception_resnet_v2',	'ens3_adv_inception_v3',	'ens4_adv_inception_v3',	'ens_adv_inception_resnet_v2', 	'resnet_v2_50',	'resnet_v2_101']

slim = tf.contrib.slim
tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')
tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')
tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', './adv/NAAs/inception_v3/', 'Output directory with images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')
tf.flags.DEFINE_integer('batch_size', 25, 'How many images process at one time.')
tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_integer('image_size',299,'the crop size')
"""parameter for DIM"""
tf.flags.DEFINE_integer('image_resize', 331, 'size of each diverse images.')
tf.flags.DEFINE_float('DI_probabilty', 1.0, 'Probability of using diverse inputs.')
"""parameter for CIM"""
tf.flags.DEFINE_integer('image_resize_crop',279,'the crop size')
"""parameter for pixel_drop"""
tf.flags.DEFINE_float('probb',0.8,'the crop size')
"""for smixup parameters"""
tf.flags.DEFINE_integer('admix_size',1 , 'Number of randomly sampled images')
tf.flags.DEFINE_float('admix_portion', 0.6, 'Number of randomly sampled images')
"""parameter for NAA"""
tf.flags.DEFINE_float('ens1', 30.0, 'Number of aggregated n.')
tf.flags.DEFINE_float('ens2', 9.0, 'Number of aggregated n.')

FLAGS = tf.flags.FLAGS

#model_names=['X101-DA','R152-B','R152-D']  network_fn.load_weight('.npy')
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #选择哪一块gpu,如果是-1，就是调用cpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# config = tf.ConfigProto()#对session进行参数配置
# config.allow_soft_placement=True # 如果你指定的设备不存在，允许TF自动分配设备
# config.gpu_options.per_process_gpu_memory_fraction=0.95#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
# config.gpu_options.allow_growth = True#按需分配显存，这个比较重要
def verify(model_name,ori_image_path,adv_image_path):
    checkpoint_path=utils.checkpoint_paths[model_name]
    #如果是防御模型，则把其转为对应的NT模型，进行统一操作
    if model_name=='adv_inception_v3' or model_name=='ens3_adv_inception_v3' or model_name=='ens4_adv_inception_v3':
        model_name='inception_v3'
    elif model_name=='adv_inception_resnet_v2' or model_name=='ens_adv_inception_resnet_v2':
        model_name='inception_resnet_v2'


    num_classes=1000+utils.offset[model_name]
    #通过网络工厂得到对应网络
    network_fn = utils.nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes),
        is_training=False)

    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    image_size = utils.image_size[model_name]

    batch_size=200
    if model_name == 'vgg_16' or model_name=='vgg_19':
        batch_size=100
    image_ph=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])

    logits, _ = network_fn(image_ph)
    predictions = tf.argmax(logits, 1)

    with tf.Session() as sess:
        #在含有tf.Variable变量下，必须使用初始化函数；其他可以不用
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.get_default_graph()
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess,checkpoint_path)

        ori_pre=[] # prediction for original images
        adv_pre=[] # prediction label for adversarial images
        ground_truth=[] # grund truth for original images
        for images,names,labels in utils.load_image_csv(ori_image_path, image_size, batch_size,
                                                        'dataset/dev_dataset.csv'):
            images=image_preprocessing_fn(images)
            #将该batch的图像集导入计算图，得到prediction。
            pres=sess.run(predictions,feed_dict={image_ph:images})
            ground_truth.extend(labels+utils.offset[model_name])#将真实label加入
            ori_pre.extend(pres)#将网络预测原图的预测值加入
        for images,names,labels in utils.load_image_csv(adv_image_path, image_size, batch_size,
                                                        'dataset/dev_dataset.csv'):
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            adv_pre.extend(pres)#将网络预测对抗样本的值加入
    #重置计算图
    tf.reset_default_graph()
    ori_pre=np.array(ori_pre)
    adv_pre=np.array(adv_pre)
    ground_truth=np.array(ground_truth)
    return ori_pre,adv_pre,ground_truth


def main(ori_path='./dataset/images/',adv_path='./adv/NAAs/inception_v3/',output_file='./log.csv'):
    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]
    argment = []
    argment.append('{:.1}'.format(FLAGS.ens2))
    argment.append('{:.1}'.format(FLAGS.admix_portion))
    #open(file_name,mode,):file_name表示打开文件名，mode表示打开模式（a+表示追加模式，直接定位在文件末尾。）
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(argment)
        #写入模型名称
        writer.writerow(model_names)
        for model_name in model_names:
            print(model_name)
            ori_pre,adv_pre,ground_truth=verify(model_name,ori_path,adv_path)
            ori_accuracy = np.sum(ori_pre == ground_truth)/1000
            adv_accuracy = np.sum(adv_pre == ground_truth)/1000
            adv_successrate = np.sum(ori_pre != adv_pre)/1000
            adv_successrate2 = np.sum(ground_truth != adv_pre) / 1000
            print('ori_acc:{:.1%}/adv_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(ori_accuracy,adv_accuracy,adv_successrate,adv_successrate2))
            ori_accuracys.append('{:.1%}'.format(ori_accuracy))
            adv_accuracys.append('{:.1%}'.format(adv_accuracy))
            adv_successrates.append('{:.1%}'.format(adv_successrate2))
        # print(adv_successrates)
        # writer.writerow(ori_accuracys)
        writer.writerow(adv_successrates)
        # writer.writerow(adv_accuracys)

"""
inception_v3
inception_v4
inception_resnet_v2
resnet_v2_152
"""
if __name__=='__main__':
    #parser=argparse.ArgumentParser()
    #parser.add_argument('--ori_path', default='./dataset/images/')
    #parser.add_argument('--adv_path',default='./adv/NAAs/inception_v3/')
    #parser.add_argument('--output_file', default='./log.csv')
    #args=parser.parse_args()
    main('./dataset/images/','./adv/NAAs/inception_v3/','./log.csv')
