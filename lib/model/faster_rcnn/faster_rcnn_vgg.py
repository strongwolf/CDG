import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.CE= torch.nn.CrossEntropyLoss()

    def forward(self, im_data, im_info, gt_boxes=None, num_boxes=None, norms=None, target=False, weighted=False):
        batch_size = im_data.size(0)
        outputs = dict()
        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat3 = self.RCNN_base3(base_feat2)
        base_feat4 = self.RCNN_base4(base_feat3)
        base_feat = self.RCNN_base5(base_feat4)
        base_feat_h, base_feat_w = base_feat.size()[-2:]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        outputs['dt_loss'] = [rpn_loss_cls, rpn_loss_bbox]
        rois = rois.detach()
        if self.training and gt_boxes is not None:
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, norms, weighted)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, weights = roi_data
                rois_label = rois_label.view(-1).long()
                rois_target = rois_target.view(-1, rois_target.size(2))
                rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
                rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
                weights = weights.view(-1)
                outputs['roi_label'] = rois_label
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
       
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        
            
        roi_feat = self._head_to_tail(pooled_feat)
        
        bbox_pred = self.RCNN_bbox_pred(roi_feat)
        cls_score = self.RCNN_cls_score(roi_feat)
        cls_prob = F.softmax(cls_score, 1)
        
        if self.training and gt_boxes is not None:
            if not self.class_agnostic:
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduction='none')
            RCNN_loss_cls = torch.mean(RCNN_loss_cls)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, reduction='none')
            RCNN_loss_bbox = torch.sum(RCNN_loss_bbox * weights / weights.sum())
            outputs['dt_loss'] += [RCNN_loss_cls, RCNN_loss_bbox]

        if not self.training:
            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
            outputs['predict'] = [rois, cls_prob, bbox_pred]
            outputs['roi_feat'] = roi_feat
            outputs['base_feat'] = base_feat
        return outputs
    

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
