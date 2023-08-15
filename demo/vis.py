# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
可视化特征提取结果
"""

import argparse
import logging
import sys
import threading

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

import torch.nn.functional as F
from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *


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
datasets_dir = 'datasets/Market-1501-v15.09.15/reid'

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

# 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。

def add_zeros(num):
    num_str = str(num)
    if len(num_str) < 2:
        return '0' * (2 - len(num_str)) + num_str
    else:
        return num_str

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


# 读取配置文件
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",  # config路径，通常包含模型配置文件
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',  # 是否并行
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",  # 数据集名字
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",  # 输出结果路径
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",  # 输出结果是否查看
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",  # 挑选多少张图像用于结果展示
        default=30,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",  # 结果展示是相似度排序方式，默认从小到大排序
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",  # label结果展示是相似度排序方式，默认从小到大排序
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",  # 显示topk的结果，默认显示前10个结果
        default=5,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
# def predict_person_id():
    args = get_parser().parse_args()
    # 调试使用，使用的时候删除下面代码
    # ---
    args.config_file = "./configs/Market1501/bagtricks_R50.yml"  # config路径
    args.dataset_name = 'Market1501'  # 数据集名字
    args.vis_label = False  # 是否显示正确label结果
    args.rank_sort = 'descending'  # 从大到小显示关联结果
    args.label_sort = 'descending'  # 从大到小显示关联结果
    # ---

    cfg = setup_cfg(args)
    # 可以直接在代码中设置cfg中设置模型路径
    # cfg["MODEL"]["WEIGHTS"] = './configs/Market1501/bagtricks_R50.yml'
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)  # 创建测试数据集
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)  # 加载特征提取器，也就是加载模型

    logger.info("Start extracting image features")
    feats = []  # 图像特征，用于保存每个行人的图像特征
    pids = []  # 行人id，用于保存每个行人的id
    camids = []  # 拍摄的摄像头，行人出现的摄像头id
    frameids=[]
    personids=[]
    # 逐张保存读入行人图像，并保存相关信息
    for (feat, pid, camid, frameid,personid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        frameids.extend(frameid)
        personids.extend(personid)
    feats = torch.cat(feats, dim=0)  # 将feats转换为tensor的二维向量，向量维度为[图像数，特征维度]
    # 这里把query和gallery数据放在一起了，需要切分query和gallery的数据
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    q_frameids = np.asarray(frameids[:num_query])
    g_frameids = np.asarray(frameids[num_query:])
    q_personids = np.asarray(personids[:num_query])
    g_personids = np.asarray(personids[num_query:])

    # compute cosine distance 计算余弦距离
    q_feat = F.normalize(q_feat, p=2, dim=1)
    g_feat = F.normalize(g_feat, p=2, dim=1)
    distmat = 1 - torch.mm(q_feat, g_feat.t())  # 这里distmat表示两张图像的距离，越小越接近
    distmat = distmat.numpy()
    print(distmat)

    # 计算各种评价指标 cmc[0]就是top1精度，应该是93%左右，这里精度会有波动
    logger.info("Computing APs for all query images ...")
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids,max_rank=args.max_rank)

    logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(test_loader.dataset)  # 创建Visualizer类
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids,q_frameids,g_frameids,q_personids,g_personids)  # 保存结果

    # logger.info("Start saving ROC curve ...")  # 保存ROC曲线
    # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    # logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")  # 保存部分查询图像的关联结果，按照顺序排列
    query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
    # createTimer()

# def createTimer():
#     global img
#     global result_position
#     # if img is not None:
#     timer = threading.Timer(5, predict_person_id)
#     timer.start()
#     print('keep going to create timer for image')
# createTimer()
# if __name__ == '__main__':
#     image_counter = 0
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     # 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
#     address = ('192.168.31.50', 8888)
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.bind(address)  # 将Socket（套接字）绑定到地址
#     s.listen(True)  # 开始监听TCP传入连接
#     print('Waiting for images...')
#     while True:
#         conn, addr = s.accept()
#
#         # 判断是否需要删除query图片和移动image图片
#         makedir(f'{datasets_dir}/image')
#         target_dir = f'{datasets_dir}/show_image'
#         makedir(target_dir)
#         makedir(f'{datasets_dir}/show_person_id')
#
#         files = os.listdir(f'{datasets_dir}/image')
#         if image_counter >= 6:
#             image_counter = 0
#             clip_files = os.listdir(f'{datasets_dir}/query_test')
#             clip_txt = os.listdir(f'{datasets_dir}/label')
#             for file in clip_files:
#                 os.remove(f'{datasets_dir}/query_test/{file}')  # 删除query_test中的图片，因为在gallery已经存放好了
#             for file in clip_txt:
#                 os.remove(f'{datasets_dir}/label/{file}')  # 删除label中的标签，因为已经切割完了
#             for file in files:
#                 shutil.move(f'{datasets_dir}/image/{file}', target_dir)  # 剪切并覆盖图片到target dir目录
#
#         filename = conn.recv(1024).decode()  # 接收图片名
#         # prefix_file=filename[:filename.index('_')]
#         print(filename)
#         prefix_file=filename
#         img_path = f'{datasets_dir}/image/{prefix_file}.jpg'
#         label_path = f'{datasets_dir}/label/{prefix_file}.txt'
#         makedir(f'{datasets_dir}/label')
#         if filename:
#             conn.send('filename finish'.encode())
#             length = recv_size(conn, 16)  # 首先接收来自客户端发送的大小信息
#             print(length.decode())
#             if isinstance(length.decode(), str):  # 若成功接收到大小信息，进一步再接收整张图片
#                 stringData = recv_size(conn, int(length))
#                 data = numpy.fromstring(stringData, dtype='uint8')
#                 decimg = cv2.imdecode(data, 1)  # 解码处理，返回mat图片
#                 cv2.imshow('SERVER', decimg)
#                 cv2.imwrite(img_path, decimg)
#                 if cv2.waitKey(10) == 27:
#                     break
#                 print('Image recieved successfully!')
#             conn.send('send result'.encode("utf-8"))
#             results = json.loads(conn.recv(1024))
#             print(results)
#             with open(label_path, 'w') as f:
#                 for i, result in enumerate(results):
#                     f.write(result + '\n')
#
#             if cv2.waitKey(10) == 27:
#                 break
#
#         # 读取图片，结果为三维数组
#
#         img = cv2.imread(img_path)
#         # 图片宽度(像素)
#         w = img.shape[1]
#         # 图片高度(像素)
#         h = img.shape[0]
#         # 打开文件，编码格式'utf-8','r+'读写
#         f = open(label_path, 'r+', encoding='utf-8')
#         # 读取txt文件中的第一行,数据类型str
#         counter = 0
#         makedir(f'{datasets_dir}/query_test')
#         makedir(f'{datasets_dir}/gallery')
#         while True:
#             line = f.readline()
#             if line != '':
#                 # line=line.replace('\n','')
#                 msg = line.split(" ")
#                 print(msg)
#                 img_roi = img[int(msg[1]):int(msg[3]), int(msg[0]):int(msg[2])]
#                 path=add_zeros(counter)
#                 save_query_path = f'{datasets_dir}/query_test/0001_{filename}_{path}.jpg'
#                 save_gallery_path = f'{datasets_dir}/gallery/0001_{filename}_{path}.jpg'
#
#                 cv2.imwrite(save_query_path, img_roi)
#                 cv2.imwrite(save_gallery_path, img_roi)
#                 counter += 1
#             else:
#                 break
#         f.close()
#         image_counter += 1
#         print(image_counter)
#         if image_counter>=6:
#             # s.close()
#             predict_person_id()
#             # conn, addr = s.accept()
#
#
#     s.close()
#     cv2.destroyAllWindows()