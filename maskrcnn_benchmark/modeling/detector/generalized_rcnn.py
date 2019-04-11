# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DensityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend_feat = [256, 256, 256, 128, 64]
        self.down_sample = nn.Conv2d(256, 256, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend = make_layers(self.backend_feat, in_channels=256, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError('not support')

    def forward(self, x):
        x = self.down_sample(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.density_head = DensityHead()
        self.density_criterion = nn.MSELoss(reduction='sum')

    def forward(self, images, target_domain_images=None, targets=None, iteration=0):
        """
        Arguments:
            iteration:
            target_domain_images:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        predication_density_map = self.density_head(features[0])
        if self.training:
            gt_density_maps = torch.stack([target.get_field('density_map').masks for target in targets], dim=0).unsqueeze(1)
            density_loss = self.density_criterion(input=predication_density_map, target=gt_density_maps) / (images.tensors.shape[0] * 2)

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {
                'density_loss': density_loss
            }
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        final_result = []
        for box_list in result:
            box_list.add_field('density_map', predication_density_map)
            final_result.append(box_list)
        return final_result
