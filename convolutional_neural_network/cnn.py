#! coding: utf-8
import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


class Conv2D(nn.Module):
    # A convolutional layer cross-correlates the input and kernel and adds a scalar bias to
    # produce an output. The two parameters of a convolutional layer are the kernel and the
    # scalar bias. When training models based on convolutional layers, we typically
    # initialize the kernels randomly, just as we would with a fully connected layer.

    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


class Conv2DWithLearning:
    def __init__(self, kernel_size, y):
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(1, 1, kernel_size=self.kernel_size, bias=False)
        self.y = y
        self.lr = 3e-2

        print(self.conv2d.bias)

        # [batch_size, channel, height, width]
    def train(self, x):
        y_hat = self.conv2d(x)
        l = (y_hat - self.y) ** 2
        self.conv2d.zero_grad()
        l.sum().backward()

        self.conv2d.weight.data[:] -= self.lr * self.conv2d.weight.grad

        return l

    def get(self):
        return self.conv2d.weight.data.reshape(self.kernel_size)


class IntegrationTest(unittest.TestCase):

    def test_object_edge_detection(self):
        shape = (6, 8)
        x = torch.ones(shape)
        x[:, 2:6] = 0
        print(x)

        # Define a kernel
        k = torch.tensor([[1.0, -1.0]])

        y = corr2d(x, k)
        print(y)
        # The 1 represents the boundary from black to white, and the -1 represents
        # the boundary from white to black.
        # tensor([[0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.]])

        # And this convolution function only could detect the vertical boundaries, and
        # the horizontal boundaries cannot be detected using this convolution function.
        y2 = corr2d(x.t(), k)

        self.assertTrue(True)

    def test_conv2d_with_learning(self):
        kernel_size = (1, 2)
        shape = (6, 8)

        x = torch.ones(shape)
        x[:, 2:6] = 0

        k = torch.tensor([[1.0, -1.0]])
        y = corr2d(x, k)

        conv2d_shape = (1, 1, 6, 8)
        x = x.reshape(conv2d_shape)
        print(y.shape)
        y = y.reshape((1, 1, 6, 7))

        net = Conv2DWithLearning(kernel_size, y)

        for epoch in range(100):
            l = net.train(x)

            if (epoch + 1) % 2 == 0:
                print(f'epoch {epoch + 1}, loss {l.sum():.3f}')

        print(net.get())

        self.assertTrue(True)

    def test_diagonal_cnn(self):
        ref = torch.ones((8, 8), dtype=torch.float32)
        # Create a lower triangular matrix with ones
        x = torch.tril(ref)
        print(x)

        k = torch.tensor([[1.0, -1.0]])

        y = corr2d(x, k)

        print(y)

        y2 = corr2d(x.t(), k)
        print(y2)

        y3 = corr2d(x.t(), k.t())
        print(y3)

        self.assertTrue(True)

    def test_convolution_paddings(self):
        # When we apply a convolutional layer is that we tend to lose pixels on the
        # perimeter of our image. One straightforward solution to this problem is to
        # add extra pixels of filler around the boundary of our input image, thus
        # increasing the effective size of the image. Typically, we set the values of
        # the extra pixels to zero.
        #
        # In many cases, we will want to set p_h = k_h - 1 and p_w = k_w - 1 to give the
        # input and output the same height and width. This will make it easier to predict
        # the output shape of each layer when constructing the network. if k_h is even,
        # one possibility is to pad ceil(p_h / 2) rows on the top of the input and
        # ceil(p_w / 2) rows on the bottom. We will pad both sides of the width in the
        # same way.
        # CNNs commonly use convolution kernels with odd height and width values,
        # such as 1, 3, 5 or 7. Choosing odd kernel sizes has the benefit that we
        # can preserve the dimensionality while padding with the same number of rows on
        # top and bottom, and the same number of columns on left and right.
        # Moreover, this practice of using odd kernels and padding to precisely preserve
        # dimensionality offers a clerical benefits. For any two-dimensional tensor x, when
        # the kernel's size is odd and the number of padding rows and columns on all sides
        # are the same, thereby producing an output with the same height and width as the
        # input, we know that the output y[i, j] is calculated by cross-correlation of the
        # input and convolution kernel with the window centered on the x[i, j].
        def comp_conv2d(conv, x):
            x = x.reshape((1, 1) + x.shape)
            y = conv(x)
            print(y.shape)
            # NOTE: y.reshape(y.shape[2:]) Use she shapes from the 3rd dimension, this because
            # that the result of the torch.nn.Conv2d operator contains the [batch_size, channels]
            # these two dimensions before the spatial shape.
            return y.reshape(y.shape[2:])

        # Applies a 2D convolution over an input signal composed of several input planes.
        # Class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        #                       padding=0, dilation=1, groups=1, bias=True,
        #                       padding_mode='zeros', device=None, dtype=None)
        # For torch.nn.Conv2d api, the `padding` controls the amount of padding
        # applied to the input. It can be either a string {'valid', 'same'} or
        # an int/a tuple of ints giving the amount of implicit padding applied
        # on BOTH SIDES.
        conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        x = torch.rand(size=(8, 8))
        res = comp_conv2d(conv2d, x)
        print(res.shape)
        # The output shape should be calculated with the following formula:
        # output.shape = (n_h - k_h + p_h + 1) * (h_w - k_w + p_w + 1)
        self.assertEqual(res.shape, x.shape)

        conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
        res = comp_conv2d(conv2d, x)
        print(res.shape)
        self.assertEqual(res.shape, x.shape)

    def test_convolution_strides(self):
        # When computing the cross-correlation, we start with the convolution window at the
        # upper-left corner of the input tensor, and then slide it over all locations both
        # down and to the right. However, sometimes, either for computational efficiency or
        # because we wish to downsample, we move our window more than one element at a time,
        # skipping the intermediate locations. This is particularly useful if the convolution
        # kernel is large since it captures a large area of the underlying image.
        # We refer to the number of rows and columns traversed per slide as STRIDE.
        def comp_conv2d(conv, x):
            x = x.reshape((1, 1) + x.shape)
            y = conv(x)
            return y.reshape(y.shape[2:])

        x = torch.rand(size=(8, 8))

        kernel_size, padding_size, stride = 3, 1, 2
        conv2d = nn.LazyConv2d(1, kernel_size=kernel_size, padding=padding_size, stride=stride)
        res = comp_conv2d(conv2d, x)
        print(res)
        print(res.shape)

        # the vertical stride is s_h, and the horizontal stride is s_h, then the output
        # shape should be calculated with the following formula:
        # output.shape = floor((n_h - k_h + p_h + s_h) / s_h) * floor((n_w - k_w + p_w + s_w) / s_w)

        output_h = int(math.floor((x.shape[0] - kernel_size + padding_size + stride) // stride))
        output_w = int(math.floor((x.shape[1] - kernel_size + padding_size + stride) // stride))

        self.assertEqual(res.shape, torch.Size([output_h, output_w]))

        channels = 1
        kernel_size = (3, 5)
        padding = (0, 1)
        stride = (3, 4)
        conv2d = nn.LazyConv2d(channels, kernel_size=kernel_size, padding=padding, stride=stride)
        res = comp_conv2d(conv2d, x)

        output_h = int(math.floor((x.shape[0] - kernel_size[0] + padding[0] + stride[0]) // stride[0]))
        output_w = int(math.floor((x.shape[1] - kernel_size[1] + padding[1] + stride[1]) // stride[1]))

        self.assertEqual(res.shape, torch.Size((output_h, output_w)))

        # Padding can increase the height and width of the output. This is often used to give
        # the output the same height and width as the input to avoid undesirable shrinkage of
        # the output. Moreover, it ensures that all pixels are used equally frequently.
        # Typically we pick symmetric padding on both sides of the input height and width. In
        # this case we refer to (p_h, p_w) padding. Most commonly we set p_h == p_w, in which case
        # we simply state that we choose padding p.
        # The stride can reduce the resolution of the output.


if __name__ == '__main__':
    unittest.main(verbosity=True)
