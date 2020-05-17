import seaborn as sns
import matplotlib.pyplot as plt
from config import sns_config
import cv2
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd


class Vis:
    def __init__(self, special_color):
        self.special_color = [c / 255. for c in special_color]
        self.cmap = sns.light_palette(self.special_color, as_cmap=True)

    def get(self, draw_function):
        sns.set()
        sns.set_style(sns_config)
        fig = plt.figure()
        draw_function(fig, self.special_color)
        buffer_ = io.BytesIO()
        fig.savefig(buffer_, format='png', bbox_inches='tight', dpi=600, pad_inches=0, transparent=True)
        buffer_.seek(0)

        img = Image.open(buffer_)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


class Word:
    def __init__(self):
        self.word_per_row = 16

    def get(self, content, color, font, bg_color):
        row_number = int(len(content) / self.word_per_row) if len(content) % self.word_per_row == 0 \
            else int(len(content) / self.word_per_row) + 1

        if row_number == 1:
            img = Image.new(mode='RGB', size=(len(content) * 100, 100), color=bg_color)
        else:
            img = Image.new(mode='RGB', size=(self.word_per_row * 50, row_number * 50), color=bg_color)
        draw = ImageDraw.Draw(img)

        if row_number == 1:
            draw.text((0, 0), content, font=ImageFont.truetype(font, 100), fill=color)
        else:
            for index in range(row_number):
                draw.text((0, index * 50), content[index * self.word_per_row:(index + 1) * self.word_per_row],
                          font=ImageFont.truetype(font, 48),
                          fill=color)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGRA)


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