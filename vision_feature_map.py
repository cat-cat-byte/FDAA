from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
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
vgg_16:'vgg_16/conv3/conv3_3/Relu'
vgg_19:'vgg_19/conv3/conv3_3/Relu'
inception_v3(bs,35,35,256):'InceptionV3/InceptionV3/Mixed_5b/concat'
inception_v4(bs,35,35,256):'InceptionV4/InceptionV4/Mixed_5b/concat'
可视化参数：
smixup:混合比例因子=[0.6,0.7]
translate:平移范围为[-10,10]
rotate:旋转范围为[-5,5]度
DI:299~331
crop:279~299
"""
tf.flags.DEFINE_string('model_name', 'vgg_19', 'The Model used to generate adv.')
tf.flags.DEFINE_string('layer_name','vgg_19/conv3/conv3_3/Relu','The layer to be attacked.')
tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')
tf.flags.DEFINE_integer('image_size',224,'the crop size')
tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')
tf.flags.DEFINE_float('probb', 0.9, 'keep probability = 1 - drop probability.')
tf.flags.DEFINE_float('ens', 30.0, 'Number of random mask input.')
tf.flags.DEFINE_float('admix_portion', 0.6, 'Number of randomly sampled images')
tf.flags.DEFINE_integer('DI_size', 330, 'size of each diverse images.')
tf.flags.DEFINE_float('DI_probabilty', 1.0, 'Probability of using diverse inputs.')
tf.flags.DEFINE_integer('image_resize_crop',279,'the crop size')
FLAGS = tf.flags.FLAGS
num = 0
def get_opt_layers(layer_name):
    opt_operations = []
    operations = tf.compat.v1.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape = op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape

def main(_):

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]#(bs,299,299,3)
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    num_classes = 1000 + utils.offset[FLAGS.model_name]
    network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
    layer_name=FLAGS.layer_name
    with tf.Graph().as_default():
        ori_input  = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        label_ph = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])
        x=tf.concat([ori_input,adv_input],axis=0)
        logits, end_points = network_fn(smixup(x,FLAGS))
        opt_operations,shape = get_opt_layers(layer_name)
        ###
        weights_tensor = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations[0])[0]
        ###

        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            saver.restore(sess,checkpoint_path)
            for images, names, labels in utils.load_image_csv(FLAGS.input_dir, FLAGS.image_size, FLAGS.batch_size,'dataset/dev_dataset.csv'):
                images_tmp = image_preprocessing_fn(np.copy(images))
                index = np.random.randint(0,FLAGS.batch_size,1)
                labels = labels + utils.offset[FLAGS.model_name]
                labels = to_categorical(np.concatenate([labels, labels], axis=-1), num_classes)

                # # # # feature weight vision
                # weight_np = np.zeros(shape=shape)
                # for l in range(int(FLAGS.ens)):
                #     x_base = np.array([0.0, 0.0, 0.0])
                #     x_base = image_preprocessing_fn(x_base)
                #     images_tmp2 = image_preprocessing_fn(np.copy(images))
                #     images_tmp2 += np.random.normal(size=images.shape, loc=0.0, scale=0.2)
                #     images_tmp2 = images_tmp2 * (1 - l / FLAGS.ens) + (l / FLAGS.ens) * x_base
                #     w, feature = sess.run([weights_tensor, opt_operations[0]],
                #                                 feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                #                                            label_ph: labels})
                #     weight_np = weight_np + w[:FLAGS.batch_size]
                # normalize the weights
                # weight_np = -normalize(weight_np, 2)[0]#(1,35,35,256)
                # weight_np = np.sum(weight_np, axis=2)
                # plt.subplot(121)
                # plt.axis('off')
                # plt.imshow(np.squeeze(images[index],axis=0))
                # plt.subplot(122)
                # plt.axis('off')
                # plt.imshow(weight_np)
                # plt.show()
                # # # #

                # # # feature map vision
                feature_base = sess.run([opt_operations[0]],feed_dict={ori_input: images_tmp, adv_input: images_tmp})
                feature_map = feature_base[0]
                feature_map = np.squeeze(feature_map[index],axis=0)
                feature_map = np.sum(feature_map, axis=2)
                feature_map = cv2.imread('./dataset/feature_map/inception_v4/'+names[index][0])[..., ::-1]

                agg_feature_base = np.zeros_like(feature_base[0])
                for i in range(int(FLAGS.ens)):
                    tmp_feature_base = sess.run(opt_operations[0],feed_dict={ori_input: images_tmp, adv_input: images_tmp})
                    agg_feature_base += tmp_feature_base
                agg_feature_base/=(FLAGS.ens)
                agg_feature_map = np.squeeze(agg_feature_base[index],axis=0)
                agg_feature_map = np.sum(agg_feature_map, axis=2)
                # #########可视化
                # plt.subplot(131)
                # plt.axis('off')
                # plt.title(names[index][0])
                # plt.imshow(np.squeeze(images[index],axis=0))
                # plt.subplot(132)
                # plt.axis('off')
                # plt.imshow(feature_map)
                # plt.subplot(133)
                # plt.axis('off')
                # plt.imshow(agg_feature_map)
                # plt.show()
                #保存图像
                plt.figure(figsize=(16, 16), dpi=100)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.imshow(agg_feature_map) #保存的对象
                #plt.savefig('/content/example.png', pad_inches=0)  # dpi=100 和上文相对应 pixel尺寸/dpi=inch尺寸
                plt.savefig('./dataset/agg_feature_map/vgg19/smixup/' + names[index][0],pad_inches=0)
                plt.clf()
                plt.close()



if __name__ == '__main__':
    tf.compat.v1.app.run()