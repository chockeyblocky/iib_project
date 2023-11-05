"""
This will contain the code for an SSIM loss function implemented in tensorflow. Implementation taken from
https://github.com/OwalnutO/SSIM-Loss-Tensroflow/blob/master/SSIM_loss_class.py and slightly modified.
"""

import tensorflow as tf
import numpy as np


class SSIM(object):
    def __init__(self, k1=0.01, k2=0.02, L=1, window_size=11):
        self.k1 = k1
        self.k2 = k2
        self.L = L  # the value range of input image pixels (i.e. max value)
        self.WS = window_size

    def _tf_fspecial_gauss(self, size, sigma=1.5):
        """Function to mimic the 'fspecial' Gaussian MATLAB function"""
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def ssim_loss(self, img1, img2):
        """
        The function is to calculate the ssim score
        """
        window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)
        # import pdb
        # pdb.set_trace()

        (_, _, _, channel) = img1.shape.as_list()

        window = tf.tile(window, [1, 1, channel, 1])

        # here we use tf.nn.depthwise_conv2d to imitate the group operation in torch.nn.conv2d
        mu1 = tf.nn.depthwise_conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        img1_2 = img1 * img1
        sigma1_sq = tf.subtract(tf.nn.depthwise_conv2d(img1_2, window, strides=[1, 1, 1, 1], padding='VALID'), mu1_sq)
        img2_2 = img2 * img2
        sigma2_sq = tf.subtract(tf.nn.depthwise_conv2d(img2_2, window, strides=[1, 1, 1, 1], padding='VALID'), mu2_sq)
        img12_2 = img1 * img2
        sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img12_2, window, strides=[1, 1, 1, 1], padding='VALID'), mu1_mu2)

        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return tf.reduce_mean(ssim_map)
