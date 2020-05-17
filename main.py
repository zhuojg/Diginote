import cv2
import os
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from canvas import Canvas, Layer
from config import sns_config
import io
from PIL import Image
import random
from vis import Vis, OneLine
import pandas as pd
from string import ascii_letters


def main():
    pass


def score_vis(fig, color, data=None):
    x = np.array(list('ABCDEF'))
    # y = random.shuffle(np.linspace(60, 100, 6))
    y = np.arange(60, 66)
    sns.barplot(x, y, palette=sns.light_palette(color, n_colors=6))


def vote_vis(fig, color, data=None):
    # Load an example dataset with long-form data
    fmri = sns.load_dataset("fmri", cache=True, data_home='./sample_data')

    # Plot the responses for different events and regions
    sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=fmri, legend=False)


def heatmap_vis(fig, color, data=None):
    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                     columns=list(ascii_letters[26:]))

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=sns.light_palette(color, as_cmap=True), vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


def scatter_vis(fig, color, data=None):
    # Load the example planets dataset
    planets = sns.load_dataset("planets", cache=True, data_home='./sample_data')

    sns.scatterplot(x="distance", y="orbital_period",
                         hue="year", size="mass",
                         palette=sns.light_palette(color, as_cmap=True), sizes=(10, 200),
                         data=planets)


def line_vis(fig, color, data=None):
    rs = np.random.RandomState(365)
    values = rs.randn(365, 4).cumsum(axis=0)
    dates = pd.date_range("1 1 2016", periods=365, freq="D")
    data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    data = data.rolling(7).mean()

    sns.lineplot(data=data, linewidth=2.5, legend=False, palette=sns.light_palette(color, n_colors=4))


def dist_vis(fig, color, data=None):
    # colors = sns.dark_palette(color, n_colors=2)

    mean, cov = [0, 2], [(1, .5), (.5, 1)]
    x, y = np.random.multivariate_normal(mean, cov, size=50).T

    # Plot a filled kernel density estimate
    # sns.distplot(data, hist=False, color=colors[0], kde_kws={"shade": True})
    sns.kdeplot(x, shade=True, color=color)

    # mean, cov = [0, 1], [(2, 1), (1.5, 1.5)]
    # x, y = np.random.multivariate_normal(mean, cov, size=50).T
    # # sns.distplot(data, hist=False, color=colors[1], kde_kws={"shade": True})
    # sns.kdeplot(x, shade=True, color=colors[1])


if __name__ == '__main__':
    background_color = (247, 237, 226)
    background_color_bgr = (226,  237, 247)
    font_color = (0, 0, 0)
    special_color = (30, 144, 255)
    # special_color = '#1E90FF'

    v = Vis(special_color=special_color)
    ol = OneLine()
    c = Canvas('framework_3')

    c.set_bg_color(background_color_bgr)
    c.set_special_color(special_color)
    c.add_layer(Layer(img=v.get(dist_vis), alpha=1, title='成绩变化'))
    c.add_layer(Layer(img=v.get(line_vis), alpha=1, title='班级成绩'))
    c.add_layer(Layer(img=v.get(score_vis), alpha=1, title='班级排名'))
    c.add_layer(Layer(img=v.get(heatmap_vis), alpha=1, title='春游报名'))
    c.add_layer(Layer(img=ol.get('语文和英语', color=c.special_color, font=c.font, bg_color=background_color), alpha=1, title='考差的科目'))
    c.add_layer(Layer(img=ol.get('7', color=c.special_color, font=c.font, bg_color=background_color), alpha=1, title='下次测验倒计时'))
    # c.add_layer(Layer(img=Vis(line_vis).get(), alpha=1))
    # c.add_layer(Layer(img=Vis(score_vis).get(), alpha=1))
    # c.add_layer(Layer(img=Vis(vote_vis).get(), alpha=1))
    # c.add_layer(Layer(img=Vis(heatmap_vis).get(), alpha=1))
    c.draw()
    c.save('result.png')
    c.show()
