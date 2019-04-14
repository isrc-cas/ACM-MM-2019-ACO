# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import glob
import os

import cv2
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def main():
    parser = argparse.ArgumentParser(description="DPNet Demo")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to images file",
    )
    parser.add_argument(
        "--save_dir",
        default='rpc_results',
        type=str,
        help="path to images file",
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
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    image_paths = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        composite = coco_demo.run_on_opencv_image(img)
        cv2.imwrite(os.path.join(args.save_dir, os.path.basename(image_path)), composite)


if __name__ == "__main__":
    main()
