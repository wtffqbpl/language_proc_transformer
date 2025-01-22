#! coding: utf-8

import numpy as np
import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline


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

