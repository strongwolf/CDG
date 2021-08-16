
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import time

import torch
import cv2
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_overlaps
from model.utils.parser_func_multi import parse_args,set_dataset_args
from lxml import etree, objectify
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def test_scl(fasterRCNN, output_dir, dataset, cfg, class_agnostic, imdb, dataset_name, roidb=None, version='default'):

    start = time.time()
    max_per_image = 100
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
    data_iter = iter(dataloader)
    num_images = len(roidb)
    _t = {'im_detect': time.time(), 'misc': time.time()}
    fasterRCNN.eval()
    num_classes = imdb.num_classes
    gt_features = [ [] for i in range(1)]
    if 'water' in dataset_name:
      feat_dim = 2048
    else:
      feat_dim = 4096
    print(num_images)
    with torch.no_grad():
        for i in range(num_images):
            print(i)
            data = next(data_iter)
            im_data = data[0].cuda()
            im_info = data[1].cuda()
            gt_boxes = data[2].cuda()
            num_boxes = data[3].cuda()
            if num_boxes == 0:
              continue
            outputs = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            base_feat = outputs['base_feat'].detach()
            rois, cls_prob, bbox_pred = outputs['predict']
            gt_boxes = gt_boxes.view(-1,5)
                    
            rois = rois.view(-1, 5)
            overlaps = bbox_overlaps(rois[:,1:], gt_boxes[:,:4])
            max_overlaps, max_inds = overlaps.max(1)
            #mask_inds = ((max_overlaps>0.00) & (max_overlaps<=0.1)).nonzero().view(-1)
            mask_inds = (max_overlaps == 0).nonzero().view(-1)
            if mask_inds.size(0) != 0 :
                rois = rois[mask_inds[0]].view(-1,5)
                roi_feat = fasterRCNN.RCNN_roi_align(base_feat, rois)
                roi_feat = fasterRCNN._head_to_tail(roi_feat)
                gt_features[0].append(roi_feat.view(-1, feat_dim).detach()) 
                    

    det_file = os.path.join(output_dir,  'bg_feats.pkl')
    print(det_file)
    for i in range(len(gt_features)):
        gt_feature = torch.cat(gt_features[i], 0).transpose(1,0).contiguous().cpu().numpy()
        gt_norm = np.linalg.norm(gt_feature, axis=0, keepdims=True)
        gt_feature /= gt_norm
        gt_features[i] = gt_feature
        print(gt_features[i].shape)
        with open(det_file, 'wb') as f:
            pickle.dump(gt_features, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  print(args.vis, "   ", args.dataset)
  # exit() 
  args = set_dataset_args(args,test=False)
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  np.random.seed(cfg.RNG_SEED)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # print('Using config:')
  # pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  from model.faster_rcnn.vgg16 import vgg16
  from model.faster_rcnn.resnet_adv import resnet

  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,lc=args.lc, gc=args.gc)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,lc=args.lc, gc=args.gc)
  else:
    print("network is not defined")

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (args.load_name))
  checkpoint = torch.load(args.load_name)
  fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  save_name = args.load_name.split('/')[-1]
  num_images = len(imdb.image_index)

  output_dir = get_output_dir(imdb, save_name)
  print(save_name, output_dir)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  test_scl(fasterRCNN, output_dir, dataset, cfg, args.class_agnostic, imdb, args.dataset, roidb=roidb)






