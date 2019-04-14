import scipy

import cv2
import glob
import json
import os
import random
from PIL import Image
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import DensityMap


# --------------------------------------------
# ----------------Test dataset----------------
# --------------------------------------------
class RPCTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.transforms = transforms
        self.images_dir = images_dir
        self.ann_file = ann_file

        with open(self.ann_file) as fid:
            data = json.load(fid)

        annotations = defaultdict(list)
        images = []
        for image in data['images']:
            images.append(image)
        for ann in data['annotations']:
            bbox = ann['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            annotations[ann['image_id']].append((ann['category_id'], x, y, w, h))

        self.images = images
        self.annotations = dict(annotations)

    def __getitem__(self, index):
        image_id = self.images[index]['id']
        img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        ann = self.annotations[image_id]
        for category, x, y, w, h in ann:
            boxes.append([x, y, x + w, y + h])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_annotation(self, image_id):
        ann = self.annotations[image_id]
        return ann

    def __len__(self):
        return len(self.images)

    def get_img_info(self, index):
        image = self.images[index]
        return {"height": image['height'], "width": image['width'], "id": image['id'], 'file_name': image['file_name']}


# --------------------------------------------
# ----------------Train dataset---------------
# --------------------------------------------
class RPCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_dir,
                 ann_file,
                 density_maps_dir=None,
                 rendered=False,
                 transforms=None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.density_maps_dir = density_maps_dir
        self.rendered = rendered
        self.transforms = transforms

        self.scale = 1.0
        self.ext = '.jpg'
        if self.rendered:  # Rendered image is 800*800 and format is png
            self.scale = 800.0 / 1815.0
            self.ext = '.png'

        with open(self.ann_file) as fid:
            self.annotations = json.load(fid)

    def __getitem__(self, index):
        ann = self.annotations[index]
        image_id = ann['image_id']
        image_name = os.path.splitext(image_id)[0]
        img_path = os.path.join(self.images_dir, image_name + self.ext)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        objects = ann['objects']
        for item in objects:
            category = item['category_id']
            x, y, w, h = item['bbox']
            boxes.append([x * self.scale, y * self.scale, (x + w) * self.scale, (y + h) * self.scale])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))

        if self.density_maps_dir:
            density_map = np.load(os.path.join(self.density_maps_dir, image_name + '.npy'))
            assert density_map.shape[0] == 200 and density_map.shape[1] == 200
            # scale to 100 * 100
            resize_scale = 2
            resize_density_map = cv2.resize(density_map,
                                            dsize=(density_map.shape[1] // resize_scale, density_map.shape[0] // resize_scale),
                                            interpolation=cv2.INTER_CUBIC) * (resize_scale ** 2)

            assert resize_density_map.shape[0] == 100 and resize_density_map.shape[1] == 100

            target.add_field('density_map', DensityMap(resize_density_map))

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotations)

    def get_img_info(self, index):
        return {"height": 1815, "width": 1815}


class RPCPseudoDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir=None, filename='pseudo_labeling.json', density=False, images_dir='test2019', annotations=None, transforms=None):
        self.root = data_dir
        self.transforms = transforms
        self.images_dir = images_dir
        self.density = density

        if annotations is not None:
            self.annotations = annotations
        else:
            self.annopath = os.path.join(self.root, filename)
            with open(self.annopath) as fid:
                annotations = json.load(fid)
            self.annotations = annotations

        print('Valid annotations: {}'.format(len(self.annotations)))

    def gaussian_filter_density(self, gt):
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density
        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
        leaf_size = 2048
        # build kd tree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leaf_size)
        # query kd tree
        distances, locations = tree.query(pts, k=4)

        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.085
                sigma = min(sigma, 999)  # avoid inf
            else:
                raise NotImplementedError('should not be here!!')
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return density

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_path = os.path.join('/data7/lufficc/rpc/test2019', ann['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        for category, x, y, w, h in ann['bbox']:
            boxes.append([x, y, x + w, y + h])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)
        scale_w = 100.0 / img.width
        scale_h = 100.0 / img.height
        if self.density:
            gt = np.zeros((100, 100))
            for category, x, y, w, h in ann['bbox']:
                cx = x + w / 2
                cy = y + h / 2
                gt[round(cy * scale_h), round(cx * scale_w)] = 1
            density = self.gaussian_filter_density(gt)
            assert density.shape[0] == 100 and density.shape[1] == 100
            target.add_field('density_map', DensityMap(density))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotations)

    def get_img_info(self, index):
        ann = self.annotations[index]
        return {"height": ann['height'], "width": ann['width'], "id": ann['id'], 'file_name': ann['file_name']}
