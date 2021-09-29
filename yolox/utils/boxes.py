#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import paddle

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = paddle.empty(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = paddle.concat((image_pred[:, :5], class_conf, class_pred.cast(paddle.float32)), 1)
        idx = paddle.masked_select(paddle.arange(0, detections.shape[0]), conf_mask)
        detections = detections[idx]
        if not detections.shape[0]:
            continue
        obj_conf = image_pred[:, 4].unsqueeze(-1)
        class_conf = image_pred[:, 5:]
        out = paddle.fluid.layers.multiclass_nms(
            bboxes=image_pred[:, :4].unsqueeze(0),
            scores=(class_conf * obj_conf).unsqueeze(0).transpose((0, 2, 1)),
            score_threshold=conf_thre,
            nms_top_k=400,
            keep_top_k=200,
            nms_threshold=nms_thre
        )
        output[i] = out

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = paddle.max(bboxes_a[:, :2], bboxes_b[:, :2])
        br = paddle.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
        area_a = paddle.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = paddle.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = paddle.maximum(
            (bboxes_a[:, :2] - bboxes_a[:, 2:] / 2).unsqueeze(1),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = paddle.minimum(
            (bboxes_a[:, :2] + bboxes_a[:, 2:] / 2).unsqueeze(1),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = paddle.prod(bboxes_a[:, 2:], 1)
        area_b = paddle.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).cast(tl.dtype).prod(axis=2)
    area_i = paddle.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a.unsqueeze(-1) + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
