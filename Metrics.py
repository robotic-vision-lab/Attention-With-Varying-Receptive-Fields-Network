import tensorflow as tf
import os
from tensorflow.keras import layers, models
import sys
from six.moves import urllib
import pdb

_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`.

    Modified from https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def lpips(input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        input0: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
        input1: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].

    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.

    Reference:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])
    # normalize to [-1, 1]
    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    input0_name, input1_name = '0:0', '1:0'

    default_graph = tf.Graph()
    producer_version = default_graph.graph_def_versions.producer

    cache_dir = os.path.expanduser('~/.lpips')
    os.makedirs(cache_dir, exist_ok=True)
    # files to try. try a specific producer version, but fallback to the version-less version (latest).
    pb_fnames = [
        '%s_%s_v%s_%d.pb' % (model, net, version, producer_version),
        '%s_%s_v%s.pb' % (model, net, version),
    ]
    for pb_fname in pb_fnames:
        if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
            try:
                _download(os.path.join(_URL, pb_fname), cache_dir)
            except urllib.error.HTTPError:
                pass
        if os.path.isfile(os.path.join(cache_dir, pb_fname)):
            break

    with open(os.path.join(cache_dir, pb_fname), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,
                                input_map={input0_name: input0, input1_name: input1})
        distance, = default_graph.get_operations()[-1].outputs

    if distance.shape.ndims == 4:
        distance = tf.squeeze(distance, axis=[-3, -2, -1])
    # reshape the leading dimensions
    distance = tf.reshape(distance, batch_shape)
    return distance
def psnr(x, y, m_val=255):
    return tf.reduce_mean(tf.image.psnr(a=x, b=y, max_val=m_val))
def ssim(x, y, m_val=255):
    return tf.reduce_mean(tf.image.ssim(img1=x, img2=y, max_val=m_val))

def ssim_ens(x, y, m_val=255):
    #yhat = tf.expand_dims(tf.reduce_mean(y,axis =1),-1)
    yhat = tf.reduce_mean(y,axis =1)
    return tf.image.ssim(img1=x, img2=yhat, max_val=m_val)

def psnr_ens(x, y, m_val=255):
    #yhat = tf.expand_dims(tf.reduce_mean(y,axis =1),-1)
    yhat = tf.reduce_mean(y,axis =1)
    return tf.image.psnr(a=x, b=yhat, max_val=m_val)
def mse(x, y):
    return tf.reduce_mean((x -y)**2)
def mae(x,y):
    return tf.reduce_mean(tf.math.abs(x-y))
def getMetricModel(input_shape):
  inputs = tf.keras.Input(input_shape)
  model = tf.keras.Model(inputs = inputs, outputs = inputs)
  return model
  
    
    
