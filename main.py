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
from vis import Vis, Word
import vis
import pandas as pd
from string import ascii_letters
import colorsys


def main():
    pass


if __name__ == '__main__':
    special_color = tuple([int(item*255) for item in colorsys.hsv_to_rgb(0.05, 0.7, 0.85)])

    # background_color = (247, 237, 226)
    # background_color_bgr = (226, 237, 247)
    background_color_bgr = background_color = (20, 20, 20)
    font_color = (240, 240, 240)
    # special_color = (88, 130, 135)

    v = Vis(special_color=special_color)
    w = Word()
    c = Canvas('framework_3')

    c.set_bg_color(background_color_bgr)
    c.set_font_color(font_color)
    c.set_special_color(special_color)
    c.add_layer(Layer(img=v.get(vis.dist_vis), alpha=1, title='成绩变化'))
    c.add_layer(Layer(img=w.get('每年一度的运动会就要开始报名了，希望大家踊跃到体育委员处报名！今年的运动会有很多新项目，班会上将给大家详细讲解。', color=c.font_color, font=c.font, bg_color=background_color), alpha=1, title='运动会报名'))
    c.add_layer(
        Layer(img=w.get('下次考试大家一定要加油！避免出现不必要的失误和粗心！', color=c.font_color, font=c.font, bg_color=background_color),
              alpha=1, title='注意事项'))
    c.add_layer(Layer(img=w.get('语文和英语', color=c.special_color, font=c.font, bg_color=background_color), alpha=1,
                      title='考差的科目'))
    c.add_layer(
        Layer(img=w.get('7', color=c.special_color, font=c.font, bg_color=background_color), alpha=1, title='下次测验倒计时'))
    c.draw()
    c.save('result.png')
    c.show()
