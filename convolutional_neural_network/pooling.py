#! coding: utf-8

import unittest
import torch
import numpy as np
import torch.nn as nn


# In many cases our ultimate task asks some global question about the image.
# Consequently, the units of our final layer should be sensitive to the entire input. By
# gradually aggregating information, yielding coarser and coarser maps, we accomplish this
# goal of ultimately learning a global representation, while keeping all advantages of
# convolutional layers at the intermediate layers of processing. The deeper we go in the
# network, the larger the receptive field (relative to the input) to which each hidden
# node is sensitive. Reducing spatial resolution accelerates this process, since the
# convolution kernels cover a larger effective area.
#
# The pooling layers, which serve the dual purpose of mitigating the sensitivity of
# convolutional layers to location and of spatially down-sampling representations.
#
# Like convolutional layers, pooling operators consist of a fixed-shape window that is
# slide over all regions in the input according to its stride, computing a single output
# for each location traversed by the fixed-shape window (sometimes known as the pooling
# window). However, unlike the cross-correlation computation of the inputs and kernels
# in the convolutional layer, the pooling layer contains no parameters (there is no
# kernel). Instead, pooling operators are deterministic, typically calculating either the
# maximum or the average value of the elements in the pooling window. These operations
# are called maximum pooling (max-pooling for short) and average pooling, respectively.
#
# The pooling layer: 它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性.

def pool2d(x, pool_size, mode='max'):
    p_h, p_w = pool_size
    y = torch.zeros((x.shape[0] - p_h + 1, x.shape[1] - p_w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode == 'max':
                y[i, j] = x[i:i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                y[i, j] = x[i:i + p_h, j: j + p_w].mean()
    return y


class IntegrationTest(unittest.TestCase):
    def test_pooling_basic(self):
        x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        res = pool2d(x, (2, 2))

        self.assertEqual(res.shape, x.shape)

    def test_pooling_module(self):
        shape = (1, 1, 4, 4)
        pooling_size = 3
        x = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
        pool = nn.MaxPool2d(pooling_size)
        res = pool(x)
        print(res)

        self.assertEqual(res.shape, (1, 1, 1, 1))
        expected = torch.tensor([[[[10.]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(res, expected))

        padding, stride = 1, 2
        pool = nn.MaxPool2d(pooling_size, padding=padding, stride=stride)
        res = pool(x)
        print(res)
        print(res.shape)

        def get_output_size(shape_, kernel_size_, padding_, stride_):
            h = np.floor((shape_[0] - kernel_size_[0] + 2 * padding_[0] + stride_[0]) // stride_[0])
            w = np.floor((shape_[1] - kernel_size_[1] + 2 * padding_[1] + stride_[1]) // stride_[1])
            return int(h), int(w)

        output_h, output_w = get_output_size(shape[2:],
                                             (pooling_size, pooling_size),
                                             (padding, padding),
                                             (stride, stride))
        output_shape = (1, 1, output_h, output_w)

        expected = torch.tensor([[[[5., 7.], [13., 15.]]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(res, expected))
        self.assertEqual(res.shape, output_shape)

        pool_size = (2, 3)
        stride = (2, 3)
        padding = (0, 1)
        pool = nn.MaxPool2d(pool_size, stride=stride, padding=padding)
        res = pool(x)
        print(res)

        output_h, output_w = get_output_size(shape[2:], pool_size, padding, stride)
        output_shape = (1, 1, output_h, output_w)

        self.assertEqual(res.shape, output_shape)

        x = torch.cat((x, x + 1), 1)
        kernel_size, padding, stride = 3, 1, 2

        pool = nn.MaxPool2d(kernel_size, padding=padding, stride=stride)


# SUMMARY:
#  Pooling is an exceeding simple operation. It does exactly what its name indicates,
# aggregate results over a window of values. All convolution semantics, such as strides
# and padding apply in the same way as they did previously. Note that pooling is
# indifferent to channels, i.e., it leaves the number of channels unchanged and it
# applies to each channel separately. Lastly, of the two popular pooling choices,
# max-pooling is preferable to average pooling, as it confers some degree of
# invariance to output. A popular choice is to pick a pooling window size of 2 x 2
# to quarter the spatial resolution of output.
# Note that there are many more ways of reducing resolution beyond pooling. For
# instance, in stochastic pooling and fractional max-pooling aggregation is combined
# with randomization. This can slightly improve the accuracy in some cases. lastly,
# as we will see later with the attention mechanism, there are more refined ways
# of aggregating over outputs, e.g., by using the alignment between a query and
# representation vectors.


if __name__ == '__main__':
    pass
