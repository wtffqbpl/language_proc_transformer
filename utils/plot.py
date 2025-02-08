#! coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib_inline import backend_inline
matplotlib.get_backend()


def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    if legend:
        axes.legend(legend)
    axes.grid()


def plot(x, y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if x has one axis
    def has_one_axis(x):
        return hasattr(x, 'ndim') and x.ndim == 1 or isinstance(x, list) and not hasattr(x[0], '__len__')

    if has_one_axis(x):
        x = [x]

    if y is None:
        x, y = [[]] * len(x), x
    elif has_one_axis(y):
        y = [y]

    if len(x) != len(y):
        x = x * len(y)
    axes.cla()

    for ax, ay, afmt in zip(x, y, fmts):
        if len(ax):
            axes.plot(ax, ay, afmt)
        else:
            axes.plot(ay, afmt)

    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()


class ImageUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def open(path=None):
        assert path is not None, "image path should not be none"
        try:
            img = Image.open(path)
        except Exception as e:
            raise RuntimeError("Could not open image")

        return img
    
    @staticmethod
    def imshow(img: Image):
        image_array = np.array(img)
        img = plt.imshow(image_array)
        plt.axis('off')
        return img

    @staticmethod
    def show_images(imgs, num_rows: int = 2, num_cols: int = 4, scale: float = 1.5):
        fig, axes = plt.subplots(num_rows, num_cols)

        for row in range(num_rows):
            for col in range(num_cols):
                linear_idx = col + row * num_cols
                new_size = (int(imgs[linear_idx].width * scale), int(imgs[linear_idx].height * scale))

                axes[row, col].imshow(imgs[linear_idx].resize(new_size))
                axes[row, col].axis('off')

        fig.show()

    @staticmethod
    def apply(img, aug, num_rows: int = 2, num_cols: int = 4, scale: float = 1.5):
        y = [aug(img) for _ in range(num_rows * num_cols)]
        ImageUtils.show_images(y, num_rows, num_cols, scale)

