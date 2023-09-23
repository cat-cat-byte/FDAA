""""""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
"""
inception_v3(bs,35,35,256):'InceptionV3/InceptionV3/Mixed_5b/concat'
inception_v4(bs,35,35,384):'InceptionV4/InceptionV4/Mixed_5e/concat'
inception_resnet_v2(bs,71,71,192):'InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu'
resnet_v2_152(bs,19,19,512):'resnet_v2_152/block2/unit_8/bottleneck_v2/add'
"""
slim = tf.contrib.slim
tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')
tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')
tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', './adv/AFMA/inception_v3/', 'Output directory with images.')
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
#os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.compat.v1.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape


def get_NAA_loss(opt_operations,weights,base_feature):
    loss = 0
    gamma = 1.0
    for layer in opt_operations:
        adv_tensor = layer[FLAGS.batch_size:]
        attribution = (adv_tensor-base_feature)*weights
        #attribution = adv_tensor*weights,测试用的是此处，但上面两者是等价的
        loss += tf.reduce_sum(attribution) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.95#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True#按需分配显存，这个比较重要

def main(_):
    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]#(20,299,299,3)
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name

    with tf.Graph().as_default():
        # Prepare graph
        ori_input  = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input],axis=0)#x.shape=(40,299,299,3)
        logits, end_points = network_fn(smixup(x,FLAGS))
        opt_operations,shape = get_opt_layers(layer_name)
        weights_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=shape)
        base_feature = tf.compat.v1.placeholder(dtype=tf.float32, shape=shape)
        #加了辅助损失无防御模型性能上升，但防御模型性能下降。
        weights_tensor = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations[0])[0]
        loss = get_NAA_loss(opt_operations,weights_ph,base_feature)
        gradient=tf.gradients(loss,adv_input)[0]

        grad_avg = tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_shape)
        accumulated_grad_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_shape)
        noise = grad_avg
        adv_input_update = adv_input
        #计算当前的图像的噪声并进行L1正则化
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        #mometum动量，accumulated_grad_ph表示累计梯度
        noise = momentum * accumulated_grad_ph + noise
        adv_input_update = adv_input_update + alpha * tf.sign(noise)

        saver=tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(config=config) as sess:
            saver.restore(sess,checkpoint_path)
            count=0
            for images,names,labels in utils.load_image_csv(FLAGS.input_dir, FLAGS.image_size, FLAGS.batch_size,
                                                            'dataset/dev_dataset.csv'):
                labels=labels+utils.offset[FLAGS.model_name]
                count+=FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)

                images_tmp=image_preprocessing_fn(np.copy(images))
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                images_adv=images
                images_adv=image_preprocessing_fn(np.copy(images_adv))
                grad_np=np.zeros(shape=batch_shape)
                weight_np = np.zeros(shape=shape)
                for i in range(num_iter):
                    if i==0:
                        for l in range(int(FLAGS.ens1)):
                            x_base = np.array([0.0,0.0,0.0])
                            x_base = image_preprocessing_fn(x_base)
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            images_tmp2 += np.random.normal(size = images.shape, loc=0.0, scale=0.2)
                            images_tmp2 = images_tmp2*(1 - l/FLAGS.ens1)+ (l/FLAGS.ens1)*x_base
                            w, feature = sess.run([weights_tensor, opt_operations[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                            weight_np = weight_np + w[:FLAGS.batch_size]
                        weight_np = -normalize(weight_np, 2)
                    images_base = np.zeros_like(images)
                    images_base = image_preprocessing_fn(images_base)

                    feature_base = sess.run([opt_operations[0]],
                                            feed_dict={ori_input: images_base, adv_input: images_base,label_ph: labels})
                    feature_base = feature_base[0][:FLAGS.batch_size]

                    #聚合特征图，对于绝大多数常见数据增强（除了添加噪声），特征图聚合等于计算梯度的均值
                    grad = np.zeros(shape=batch_shape,dtype=np.float32)
                    for l in range(int(FLAGS.ens2)):
                        grad += sess.run(gradient,feed_dict={ori_input: images_tmp, adv_input: images_adv, weights_ph: weight_np,base_feature: feature_base,label_ph: labels})
                    grad /=FLAGS.ens2
                    images_adv, grad_np = sess.run([adv_input_update, noise],
                                                                     feed_dict={ori_input: images_tmp,
                                                                                adv_input: images_adv,
                                                                                weights_ph: weight_np,
                                                                                base_feature: feature_base,
                                                                                label_ph: labels,
                                                                                accumulated_grad_ph: grad_np,
                                                                                grad_avg:grad})

                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)
                images_adv = inv_image_preprocessing_fn(images_adv)
                utils.save_image(images_adv, names, FLAGS.output_dir)

if __name__ == '__main__':
    print(FLAGS.ens2,FLAGS.admix_portion)
    tf.compat.v1.app.run()