import os
import re
import threading
from multiprocessing import Pool

import time
import random

import cv2

# function()
filename = '0000_c1s1_000021_00.jpg'


print(int(filename[-13:-7]))
flag=False
# with open('file.txt', 'r+') as file:
#     lines = file.readlines()
#     line_num = 1
#     line = lines[line_num]
#     msg = line.split(" ")
#     if len(msg)==5:
#         person_id=msg[4]
#         flag=True
#     else:
#         line = line.replace('\n', '') + ' 新信息\n'
#
#     lines[line_num] = line
#     file.seek(0)
#     file.writelines(lines)
# print(person_id if flag else 'haha')
# with open('file.txt') as f:
#     for line in f:
#         print(line)

# label_list=[d for d in os.listdir('../datasets/Market-1501-v15.09.15/reid/label') if '.txt' in d]
# for label in label_list:
#     save_image_path=f'{label[:4]}'+'_000001.jpg'
#     image_raw=cv2.imread(f'../datasets/Market-1501-v15.09.15/reid/image/{save_image_path}')
#     with open(f'../datasets/Market-1501-v15.09.15/reid/label/{label}','r+') as file:
#         for line in file:
#             msg = line.split(" ")
#             box=[msg[0],msg[1],msg[2],msg[3]]
#             print(msg[4][:-1])
# import sys
# from PyQt5.QtWidgets import QWidget, QApplication
#
# app = QApplication(sys.argv)
# widget = QWidget()
# widget.resize(640, 480)
# widget.setWindowTitle("Hello, PyQt5!")
# widget.show()
# sys.exit(app.exec())
# s = "abc_1234_haha"
# print(s[:s.index('_')])
#
# def add_zeros(num):
#     num_str = str(num)
#     if len(num_str) < 2:
#         return '0' * (2 - len(num_str)) + num_str
#     else:
#         return num_str
# print(add_zeros(1))
def plot_one_box( x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

label_list=[d for d in os.listdir('../datasets/Market-1501-v15.09.15/reid/label') if '.txt' in d]
for label in label_list:
    save_image_path = f'{label[:-4]}.jpg'
    image_raw = cv2.imread(f'../datasets/Market-1501-v15.09.15/reid/image/{save_image_path}')
    with open(f'../datasets/Market-1501-v15.09.15/reid/label/{label}', 'r+') as file:
        for line in file:
            if line == '\n': continue
            msg = line.split(" ")

            box = [msg[0], msg[1], msg[2], msg[3]]
            print(msg)
            plot_one_box(
                box,
                image_raw,
                label="person id {}".format(
                    msg[4][:-1]
                ),
            )
    print(save_image_path)
    cv2.imwrite(f'../datasets/Market-1501-v15.09.15/reid/show_person_id/{save_image_path}', image_raw)