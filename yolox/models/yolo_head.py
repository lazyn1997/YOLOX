#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Layer):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        self.stems = nn.LayerList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [paddle.zeros([1])] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.reshape([self.n_anchors, -1])
            paddle.full(b.shape, -math.log((1 - prior_prob) / prior_prob))
            conv.bias.set_value(b.reshape([-1]))
        for conv in self.obj_preds:
            b = conv.bias.reshape([self.n_anchors, -1])
            paddle.full(b.shape, -math.log((1 - prior_prob) / prior_prob))
            conv.bias.set_value(b.reshape([-1]))

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = paddle.concat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    paddle.full([1, grid.shape[1]],
                                fill_value=stride_this_level,
                                dtype=xin[0].dtype))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.reshape(
                        [batch_size, self.n_anchors, 4, hsize, wsize]
                    )
                    reg_output = reg_output.transpose([0, 1, 3, 4, 2]).reshape(
                        [batch_size, -1, 4]
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = paddle.concat(
                    [reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                paddle.concat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = paddle.concat(
                [x.flatten(start_axis=2) for x in outputs], axis=2
            ).transpose([0, 2, 1])
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2).reshape([1, 1, hsize, wsize, 2]).cast(dtype)
            self.grids[k] = grid

        output = output.reshape([batch_size, self.n_anchors, n_ch, hsize, wsize])
        output = output.transpose([0, 1, 3, 4, 2]).reshape(
            [batch_size, self.n_anchors * hsize * wsize, -1]
        )
        grid = grid.reshape([1, -1, 2])
        xy = (output[:, :, :2] + grid) * stride
        wh = paddle.exp(output[:, :, 2:4]) * stride
        obj_cls = output[:, :, 4:]
        output = paddle.concat([xy, wh, obj_cls], 2)
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2).reshape([1, -1, 2])
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(paddle.full((*shape, 1), stride))

        grids = paddle.concat(grids, axis=1).cast(dtype)
        strides = paddle.concat(strides, axis=1).cast(dtype)

        xys = (outputs[:, :, :2] + grids) * strides
        whs = paddle.exp(outputs[:, :, 2:4]) * strides
        obj_clss = outputs[:, :, 4:]
        outputs = paddle.concat([xys, whs, obj_clss], 2)
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[:, :, :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(axis=2) > 0).cast(paddle.int32).sum(axis=1)  # number of objects
        total_num_anchors = outputs.shape[1]
        x_shifts = paddle.concat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = paddle.concat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = paddle.concat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                newtype = outputs.dtype
                cls_target = paddle.zeros([0, self.num_classes], dtype=newtype)
                reg_target = paddle.zeros([0, 4], dtype=newtype)
                l1_target = paddle.zeros([0, 4], dtype=newtype)
                obj_target = paddle.zeros([total_num_anchors, 1], dtype=newtype)
                fg_mask = paddle.zeros([total_num_anchors]).astype(bool)
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.cast(paddle.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image.gather(matched_gt_inds)
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        paddle.zeros([num_fg_img, 4]),
                        gt_bboxes_per_image.gather(matched_gt_inds),
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.cast(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = paddle.concat(cls_targets, 0)
        reg_targets = paddle.concat(reg_targets, 0)
        obj_targets = paddle.concat(obj_targets, 0)
        fg_masks = paddle.concat(fg_masks, 0)
        if self.use_l1:
            l1_targets = paddle.concat(l1_targets, 0)

        fg_masks_idx = fg_masks.nonzero()
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(paddle.gather(
            bbox_preds.reshape([-1, 4]), fg_masks_idx),
            reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.reshape(
            [-1, 1]), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(paddle.gather(
            cls_preds.reshape([-1, self.num_classes]), fg_masks_idx),
            cls_targets)).sum() / num_fg
        if self.use_l1:
            loss_l1 = (self.l1_loss(paddle.gather(
                origin_preds.reshape([-1, 4]), fg_masks_idx),
                l1_targets)).sum() / num_fg
        else:
            loss_l1 = paddle.to_tensor(0.0)

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = paddle.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = paddle.log(gt[:, 3] / stride + eps)
        return l1_target

    @paddle.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().astype("float32")
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().astype("float32")
            gt_classes = gt_classes.cpu().astype("float32")
            expanded_strides = expanded_strides.cpu().astype("float32")
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        fg_mask_idx = fg_mask.nonzero()
        bboxes_preds_per_image = paddle.gather(bboxes_preds_per_image, fg_mask_idx, axis=0)
        cls_preds_ = paddle.gather(cls_preds[batch_idx], fg_mask_idx, axis=0)
        obj_preds_ = paddle.gather(obj_preds[batch_idx], fg_mask_idx, axis=0)

        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            paddle.tile(F.one_hot(paddle.cast(gt_classes, paddle.int64), self.num_classes).astype("float32")
                        .unsqueeze(1), [1, num_in_boxes_anchor, 1])
        )

        pair_wise_ious_loss = -paddle.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with paddle.amp.auto_cast(enable=False):
            cls_preds_ = (
                    F.sigmoid(paddle.tile(cls_preds_.cast(paddle.float32).unsqueeze(0), [num_gt, 1, 1]))
                    * F.sigmoid(paddle.tile(obj_preds_.unsqueeze(0), [num_gt, 1, 1]))
            )
            pair_wise_cls_loss = paddle.sum(
                F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"),
                axis=-1)

        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (paddle.logical_not(is_in_boxes_and_center))
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile(repeat_times=[num_gt, 1])
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile(repeat_times=[num_gt, 1])
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile(repeat_times=[1, total_num_anchors])
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile(repeat_times=[1, total_num_anchors])
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile(repeat_times=[1, total_num_anchors])
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile(repeat_times=[1, total_num_anchors])
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = paddle.min(bbox_deltas, axis=-1) > 0.0
        is_in_boxes_all = paddle.sum(is_in_boxes.astype("int"), axis=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            [1, total_num_anchors]) - center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            [1, total_num_anchors]) + center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            [1, total_num_anchors]) - center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            [1, total_num_anchors]) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = paddle.min(center_deltas, axis=-1) > 0.0
        is_in_centers_all = paddle.sum(is_in_centers.astype("int"), axis=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all.logical_or(is_in_centers_all)
        idx = is_in_boxes_anchor.nonzero()

        is_in_boxes_and_center = (
            is_in_boxes.astype("int").gather(idx, axis=1).astype("bool").logical_and(
                is_in_centers.astype("int").gather(idx, axis=1).astype("bool"))

        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = paddle.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = paddle.topk(ious_in_boxes_matrix, n_candidate_k, axis=1)
        dynamic_ks = paddle.clip(topk_ious.sum(1).astype("int64"), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx, pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = paddle.sum(matching_matrix, 0)
        if paddle.sum((anchor_matching_gt > 1).astype("int32")) > 0:
            idx = (anchor_matching_gt > 1).nonzero()
            cost_argmin = paddle.argmin(cost.gather(idx, axis=1), axis=0)
            matching_matrix = matching_matrix.numpy()
            matching_matrix[:, idx] *= 0.0
            matching_matrix[cost_argmin, idx] = 1.0

        matching_matrix = paddle.to_tensor(matching_matrix)

        fg_mask_inboxes = paddle.sum(matching_matrix, 0) > 0.0
        num_fg = paddle.sum(fg_mask_inboxes.astype("int")).item()
        idx = fg_mask_inboxes.nonzero()
        fg_mask[fg_mask.clone().nonzero().numpy()[:, 0]] = fg_mask_inboxes

        matched_gt_inds = paddle.gather(matching_matrix, idx, axis=1).argmax(0)
        gt_matched_classes = paddle.gather(gt_classes, matched_gt_inds)

        pred_ious_this_matching = paddle.gather((matching_matrix * pair_wise_ious).sum(0), idx)
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
