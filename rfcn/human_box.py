#!encoding:utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import _init_paths
import pdb
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
#  update_config(cur_path + '/../experiments/rfcn/cfgs/rfcn_coco_demo.yaml')
update_config(cur_path + '/../experiments/rfcn/cfgs/resnet_v1_101_coco_trainval_rfcn_dcn_end2end_ohem.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')
    parser.add_argument('--img-dir', type=str, default='/home/xyliu/Videos/sports/dance', help='input image directory for detection')
    args = parser.parse_args()
    return args

args = parse_args()

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # load demo data
    #  image_names = ['COCO_test2015_000000000891.jpg', 'COCO_test2015_000000001669.jpg']
    image_dir = args.img_dir
    for img_dir,_, names_ in os.walk(image_dir):
        image_names = names_
    img_names = [i for i in image_names if 'jpg' in i]
    img_names.sort()

    data = []
    for im_name in tqdm(img_names):
        print(im_name)
        try:
            im = cv2.imread(img_dir+'/'+im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})
        except:
            import pdb;pdb.set_trace()
            print('--------')


    import pdb;pdb.set_trace()
    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)

    # test

    # human box
    import pdb;pdb.set_trace()
    human_boxes = []
    #  for idx, im_name in tqdm(enumerate(img_names)):
    for idx, im_name in enumerate(tqdm(img_names)):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        im = cv2.imread(img_dir +'/'+ im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        #######################################
        ### dets_nms[0] 表示所有person bboxes
        img_name = os.path.join(img_dir, im_name)
        img_box = list(dets_nms[0][0][:4])
        img_box = [round(i) for i in img_box]
        human_boxes.append({'name':img_name, 'bbox':img_box})

        #  import pdb;pdb.set_trace()
        #  show_boxes(im, dets_nms, classes, 1)

    ############ save to json
    #  import pdb;pdb.set_trace()
    import json
    data = json.dumps(human_boxes)
    json_file = img_dir +'/det.json'
    with open(json_file, 'wt') as f:
        f.write(data)

    print 'json file save in '.format(json_file)

if __name__ == '__main__':
    main()
