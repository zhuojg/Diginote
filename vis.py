import seaborn as sns
import matplotlib.pyplot as plt
from config import sns_config
import cv2
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np


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


class OneLine:
    def __init__(self):
        pass

    def get(self, content, color, font, bg_color):
        assert len(content) <= 6
        img = Image.new(mode='RGB', size=(len(content) * 100, 100), color=bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), content, font=ImageFont.truetype(font, 100), fill=color)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGRA)
