# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import glob
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image",
        metavar="FILE",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    if os.path.exists('rpc_demo_results'):
        shutil.rmtree('rpc_demo_results')
    os.mkdir('rpc_demo_results')

    with open('/data7/lufficc/rpc/instances_test2019.json') as fid:
        data = json.load(fid)

    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = defaultdict(list)
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']].append(x)
    annotations = dict(annotations)

    counter = {
        'easy': 0,
        'medium': 0,
        'hard': 0,
    }
    data_images = data['images'].copy()
    random.shuffle(data_images)
    for image_ann in data_images:
        if counter[image_ann['level']] > 10:
            continue
        image_path = os.path.join(args.image, image_ann['file_name'])
        img = cv2.imread(image_path)
        annotation = annotations[image_ann['file_name']]
        composite, correct = coco_demo.run_on_opencv_image(img, annotation)
        if correct:
            print('Get {}.'.format(image_ann['level']))
            cv2.imwrite(os.path.join('rpc_demo_results', image_ann['level'] + '_' + os.path.basename(image_path)), composite)
            counter[image_ann['level']] += 1

    # start_time = time.time()
    # for image_path in tqdm(glob.glob(os.path.join(args.image, '*.jpg'))):
    #     img = cv2.imread(image_path)
    #     composite = coco_demo.run_on_opencv_image(img)
    #     cv2.imwrite(os.path.join('rpc_demo_results', os.path.basename(image_path)), composite)
    # print("Time: {:.2f} s / img".format(time.time() - start_time))


if __name__ == "__main__":
    main()
