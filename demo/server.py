#!/usr/bin/python
# -*-coding:utf-8 -*-
import json
import os
import socket
import struct
import time

import cv2
import numpy
import shutil
# from vis import predict_person_id

# 接受图片大小的信息
datasets_dir = '../datasets/Market-1501-v15.09.15/reid'

def makedir(filepath):
    if not os.path.exists(filepath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
makedir(datasets_dir)

def recv_size(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


# socket.AF_INET 用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM 代表基于TCP的流式socket通信
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
address = ('10.28.168.215', 6666)
s.bind(address)  # 将Socket（套接字）绑定到地址
s.listen(True)  # 开始监听TCP传入连接
print('Waiting for images...')

# 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。

def add_zeros(num):
    num_str = str(num)
    if len(num_str) < 2:
        return '0' * (2 - len(num_str)) + num_str
    else:
        return num_str

image_counter = 0
while True:
    conn, addr = s.accept()

    # 判断是否需要删除query图片和移动image图片
    makedir(f'{datasets_dir}/image')
    target_dir = f'{datasets_dir}/show_image'
    makedir(target_dir)
    makedir(f'{datasets_dir}/show_person_id')

    files = os.listdir(f'{datasets_dir}/image')
    if image_counter >= 6:
        image_counter = 0
        clip_files = os.listdir(f'{datasets_dir}/query_test')
        clip_txt = os.listdir(f'{datasets_dir}/label')
        for file in clip_files:
            os.remove(f'{datasets_dir}/query_test/{file}')  # 删除query_test中的图片，因为在gallery已经存放好了
        for file in clip_txt:
            os.remove(f'{datasets_dir}/label/{file}')  # 删除label中的标签，因为已经切割完了
        for file in files:
            shutil.move(f'{datasets_dir}/image/{file}', target_dir)  # 剪切并覆盖图片到target dir目录

    filename = conn.recv(1024).decode()  # 接收图片名
    # prefix_file=filename[:filename.index('_')]
    prefix_file=filename
    img_path = f'{datasets_dir}/image/{prefix_file}.jpg'
    label_path = f'{datasets_dir}/label/{prefix_file}.txt'
    makedir(f'{datasets_dir}/label')
    if filename:
        conn.send('filename finish'.encode())
        length = recv_size(conn, 16)  # 首先接收来自客户端发送的大小信息
        print(length.decode())
        if isinstance(length.decode(), str):  # 若成功接收到大小信息，进一步再接收整张图片
            stringData = recv_size(conn, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')
            decimg = cv2.imdecode(data, 1)  # 解码处理，返回mat图片
            cv2.imshow('SERVER', decimg)
            cv2.imwrite(img_path, decimg)
            if cv2.waitKey(10) == 27:
                break
            print('Image recieved successfully!')
        conn.send('send result'.encode("utf-8"))
        results = json.loads(conn.recv(1024))
        print(results)
        with open(label_path, 'w') as f:
            for i, result in enumerate(results):
                f.write(result + '\n')

        if cv2.waitKey(10) == 27:
            break

    # 读取图片，结果为三维数组

    img = cv2.imread(img_path)
    # 图片宽度(像素)
    w = img.shape[1]
    # 图片高度(像素)
    h = img.shape[0]
    # 打开文件，编码格式'utf-8','r+'读写
    f = open(label_path, 'r+', encoding='utf-8')
    # 读取txt文件中的第一行,数据类型str
    counter = 0
    makedir(f'{datasets_dir}/query_test')
    makedir(f'{datasets_dir}/gallery')
    while True:
        line = f.readline()
        if line != '':
            # line=line.replace('\n','')
            msg = line.split(" ")
            print(msg)
            img_roi = img[int(msg[1]):int(msg[3]), int(msg[0]):int(msg[2])]
            path=add_zeros(counter)
            save_query_path = f'{datasets_dir}/query_test/0000_{filename}_{path}.jpg'
            save_gallery_path = f'{datasets_dir}/gallery/0000_{filename}_{path}.jpg'

            cv2.imwrite(save_query_path, img_roi)
            cv2.imwrite(save_gallery_path, img_roi)
            counter += 1
        else:
            break
    image_counter += 1

    f.close()

s.close()
cv2.destroyAllWindows()
