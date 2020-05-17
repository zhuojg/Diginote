import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import seaborn as sns


class Canvas:
    def __init__(self, framework):
        with open('framework.json', 'r') as f:
            self.framework = json.load(f)

        self.framework = self.framework[framework]
        self.width = self.framework['size'][0]
        self.height = self.framework['size'][1]
        self.canvas = np.zeros((self.height, self.width, 3), dtype='uint8')
        self.bg_color = (200, 200, 200)
        self.special_color = (0, 0, 0)
        self.font_color = (0, 0, 0)

        self.layers = []
        self.title = '班级公告'
        self.title_font = 'C:\\Users\\jing\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZDBSJW.TTF'

        self.sub_title_size = self.framework['subtitle_size'][0]
        self.sub_title_margin_top = self.framework['subtitle_size'][1]
        self.sub_title_margin_bottom = self.framework['subtitle_size'][2]
        self.font = 'C:\\Users\\jing\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZDBSJW.TTF'

    def set_bg_color(self, color):
        self.bg_color = color

    def set_special_color(self, color):
        self.special_color = color

    def set_font_color(self, color):
        self.font_color = color

    def set_title(self, title):
        self.title = title

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_text(self, canvas, text, pos, color, font, font_size):
        if canvas.shape[2] == 4:
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA))
        else:
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=ImageFont.truetype(font, font_size), fill=color)

        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    def save(self, path):
        cv2.imwrite(path, self.canvas)

    def show(self):
        cv2.imshow('Canvas', self.canvas)
        cv2.waitKey(0)

    def draw(self):
        canvas = np.zeros((self.height, self.width, 3), dtype='uint8')

        # set background
        for i, row in enumerate(canvas):
            for j, col in enumerate(row):
                canvas[i][j] = list(self.bg_color)

        # add title
        pos = (self.framework['title'][0], self.framework['title'][1])
        font_size = self.framework['title'][2]
        canvas = self.add_text(canvas, self.title, pos=pos, color=tuple(self.special_color),
                               font=self.title_font,
                               font_size=font_size)

        # add each layer on canvas
        for index, layer in enumerate(self.layers):
            # scale img to framework size
            element_size = self.framework['element_%d' % index]
            element_height = element_size[3] - element_size[1]
            element_width = element_size[2] - element_size[0]
            if layer.title is not None:
                element_height -= (self.sub_title_size + self.sub_title_margin_top + self.sub_title_margin_bottom * 2)
            scale = min(element_width / layer.img.shape[1], element_height / layer.img.shape[0])
            temp_img = cv2.resize(layer.img, None, fx=scale, fy=scale)

            # add subtitle of img
            if layer.title is not None:
                sub_title_height = self.sub_title_size + self.sub_title_margin_top + self.sub_title_margin_bottom * 2
                temp_canvas = np.zeros((element_size[3] - element_size[1], element_width, 3), dtype='uint8')

                for i, row in enumerate(temp_canvas):
                    for j, col in enumerate(row):
                        temp_canvas[i][j] = list(self.bg_color)

                temp_canvas = self.add_text(temp_canvas, layer.title,
                                            pos=(0, self.sub_title_margin_top),
                                            color=tuple(self.font_color),
                                            font=self.font,
                                            font_size=self.sub_title_size
                                            )
                temp_canvas = cv2.line(temp_canvas,
                                       (0, self.sub_title_margin_top + self.sub_title_size + self.sub_title_margin_bottom),
                                       (element_width, self.sub_title_margin_top + self.sub_title_size + self.sub_title_margin_bottom),
                                       color=self.font_color, thickness=1)
                temp_canvas = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2BGRA)
                width_offset = int((temp_canvas.shape[1] - temp_img.shape[1]) / 2)
                height_offset = sub_title_height
                temp_canvas[height_offset:height_offset + temp_img.shape[0], width_offset:width_offset + temp_img.shape[1]] = temp_img
                # temp_img = cv2.vconcat([temp_canvas, temp_img])
                temp_img = temp_canvas

            layer.y = int(element_size[1] + (element_height - temp_img.shape[0]) / 2)
            layer.x = int(element_size[0] + (element_width - temp_img.shape[1]) / 2)

            # if img have alpha channel, process it pixel by pixel
            if temp_img.shape[2] != 4:
                roi = canvas[layer.y:layer.y + temp_img.shape[0], layer.x:layer.x + temp_img.shape[1]]
                canvas[layer.y:layer.y + temp_img.shape[0], layer.x:layer.x + temp_img.shape[1]] = \
                    cv2.addWeighted(temp_img, layer.alpha, roi, 1 - layer.alpha, 0)
            else:
                    for i, row in enumerate(temp_img):
                        for j, col in enumerate(row):
                            try:
                                temp_alpha = col[3] / 255. * layer.alpha
                                temp = \
                                    cv2.addWeighted(canvas[layer.y + i, layer.x + j], 1 - temp_alpha, col[:3], temp_alpha, 0)
                                temp = np.reshape(temp, [3])
                                canvas[layer.y + i, layer.x + j] = temp
                            except Exception as e:
                                print(layer.y + i, layer.x + j)
                                print(canvas.shape)
                                print(temp_img.shape)
                                print(row.shape)
                                print(layer.x)
                                print(layer.y)
                                exit()

        self.canvas = canvas


class Layer:
    def __init__(self, img, alpha, title=None):
        self.img = img
        self.x = 0
        self.y = 0
        self.alpha = alpha
        self.title = title
