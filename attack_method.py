from utils import *

slim = tf.contrib.slim
"""
tf.random_uniform(shape,minval=,maxval=,dtype=)从均匀分布中采样出一个形状为shape，范围为[minval,maxval)，值的类型为dtype的张量。
"""

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2
def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0, 0], [kern_size, kern_size], [kern_size, kern_size], [0, 0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    return x
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


def input_diversity(input_tensor,FLAGS):
  rnd = tf.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = FLAGS.image_resize - rnd
  w_rem = FLAGS.image_resize - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
  return padded
  #return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.DI_probabilty), lambda: padded, lambda: input_tensor)

def rotate(input_tensor):
    rotated = tf.contrib.image.rotate(
        input_tensor,
        tf.random_uniform((), minval=-np.pi / 36, maxval=np.pi / 36),  # 10
        interpolation='NEAREST',
        name=None
    )
    return rotated

def translate(input_tensor):
    scale=10
    dx = tf.random_uniform((),minval=-scale,maxval=scale,dtype=tf.int32)
    dy = tf.random_uniform((),minval=-scale,maxval=scale,dtype=tf.int32)
    translated = tf.contrib.image.translate(
        input_tensor,
        [dx,dy],
        interpolation='NEAREST',
        name=None
    )
    return translated

def crop(input_tensor,FLAGS):
    rnd = tf.random_uniform((), FLAGS.image_resize_crop, FLAGS.image_size, dtype=tf.int32)
    crop = tf.image.resize_with_crop_or_pad(input_tensor,rnd,rnd)
    h_rem = FLAGS.image_size - rnd
    w_rem = FLAGS.image_size - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(crop, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_size, FLAGS.image_size, 3))
    return padded

def admix(input_tensor,FLAGS):
    indices = tf.range(start=0, limit=tf.shape(input_tensor)[0], dtype=tf.int32)
    return tf.concat([(input_tensor + FLAGS.admix_portion * tf.gather(input_tensor, tf.random.shuffle(indices))) for _ in range(FLAGS.admix_size)], axis=0)

def smixup(input_tensor,FLAGS):
    return (1-FLAGS.admix_portion)*input_tensor+FLAGS.admix_portion*translate(input_tensor)

def random(input_tensor):
    pass

def Scale(input_tensor):
    batch = tf.concat([input_tensor, input_tensor/2., input_tensor/4., input_tensor/8., input_tensor/16.], axis=0)
    return batch

def pixel_drop(input_tensor,FLAGS):
    mask = np.random.binomial(1, FLAGS.probb, size=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    ret = input_tensor * mask
    return ret










def transformImg(imgIn,forward_transform):
    t = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
    # please notice that forward_transform must be a float matrix,
    # e.g. [[2.0,0,0],[0,1.0,0],[0,0,1]] will work
    # but [[2,0,0],[0,1,0],[0,0,1]] will not
    imgOut = tf.contrib.image.transform(imgIn, t, interpolation="NEAREST",name=None)
    return imgOut



def batch_flip(batch):
    fun = lambda x: tf.image.random_flip_left_right(x, seed=None)
    return tf.map_fn(fun, batch)

def norm_li(a, b):
    # return torch.max(torch.abs(torch.sub(a, b)))
    return tf.abs(tf.subtract(a, b))

