import os
import numpy as np
import pprint
import pdb
import time
import _init_paths
import math
import logging

import torch
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import adjust_learning_rate, save_checkpoint,  sampler
from model.utils.parser_func_multi import parse_args, set_dataset_args
from test_net import test_scl
if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    imdb_tv, roidb_tv, ratio_list_tv, ratio_index_tv = combined_roidb(args.imdbval_name_target,False)

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + '2' + args.dataset_t +'/' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers, drop_last=True)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, 1,
                                               sampler=sampler_batch_t, num_workers=args.num_workers, drop_last=True)
    dataset_tv = roibatchLoader(roidb_tv, ratio_list_tv, ratio_index_tv, 1, \
                               imdb.num_classes, training=False, normalize=False)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()
    best_score = 0
    if args.resume:
        checkpoint = torch.load(args.load_name)
        fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
        best_score = checkpoint['best_score']
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        
    iters_per_epoch = min(len(dataloader_s), len(dataloader_t))
    args.disp_interval = 10
    count_iter = 0

    if args.evaluate:
        save_name = os.path.join(output_dir,'model_best.pth')
        test_output_dir = get_output_dir(imdb_tv, save_name.split('/')[-1])
        aps = test_scl(fasterRCNN, test_output_dir, dataset_tv, cfg, args.class_agnostic, imdb_tv, args.dataset_t)
        s = ''
        for ap in aps:
            s += str(ap) + ' ' 
        s += str(np.mean(aps))

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        
        # setting to train mode
        fasterRCNN.train()
        s_loss_temp = 0
        s_loss_rpn_cls = 0
        s_loss_rpn_box = 0
        s_loss_rcnn_cls = 0
        s_loss_rcnn_box = 0
        start = time.time()

        if epoch-1 in args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            count_iter += 1
            if args.cuda:
                im_data = data_s[0].cuda()
                im_info = data_s[1].cuda()
                gt_boxes = data_s[2].cuda()
                num_boxes = data_s[3].cuda()
                target_im_data = data_t[0].cuda()
                target_im_info = data_t[1].cuda()
                target_gt_boxes = data_t[2].cuda()
                target_num_boxes = data_t[3].cuda()
                target_norms = data_t[4].cuda()
            fasterRCNN.zero_grad()
            outputs = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            
            s_rpn_loss_cls,s_rpn_loss_box, s_RCNN_loss_cls, s_RCNN_loss_box  = outputs['dt_loss']
            s_rois_label = outputs['roi_label']
            s_loss = s_rpn_loss_cls.mean() +  s_rpn_loss_box.mean() \
                   + s_RCNN_loss_cls.mean() + s_RCNN_loss_box.mean()
            s_loss_temp += s_loss.item()
            s_loss_rpn_cls += s_rpn_loss_cls.mean().item()
            s_loss_rpn_box += s_rpn_loss_box.mean().item()
            s_loss_rcnn_cls += s_RCNN_loss_cls.mean().item()
            s_loss_rcnn_box += s_RCNN_loss_box.mean().item()
            
            outputs = fasterRCNN(target_im_data, target_im_info, target_gt_boxes, target_num_boxes, target_norms, target=True, weighted=True)
            rois_label = outputs['roi_label']
            t_rpn_loss_cls,t_rpn_loss_box, t_RCNN_loss_cls, t_RCNN_loss_box  = outputs['dt_loss']
            t_loss = t_rpn_loss_cls.mean() + t_rpn_loss_box.mean() \
                   + t_RCNN_loss_cls.mean() + t_RCNN_loss_box.mean()
            
            loss = 2*s_loss + 1*t_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    s_loss_temp /= (args.disp_interval + 1)
                    
                    s_loss_rpn_cls /= (args.disp_interval + 1)
                    s_loss_rpn_box /= (args.disp_interval + 1)
                    s_loss_rcnn_cls /= (args.disp_interval + 1)
                    s_loss_rcnn_box /= (args.disp_interval + 1)

                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] s_loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, s_loss_temp, lr))
                print("\t fg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                
                print(
                    "\t s_rpn_cls: %.4f, s_rpn_box: %.4f, s_rcnn_cls: %.4f, s_rcnn_box %.4f" \
                    % (s_rpn_loss_cls.mean().item(), s_rpn_loss_box.mean().item(), s_RCNN_loss_cls.mean().item(), s_RCNN_loss_box.mean().item()))
                print(
                    "\t t_rpn_cls: %.4f, t_rpn_box: %.4f, t_rcnn_cls: %.4f, t_rcnn_box %.4f" \
                    % (t_rpn_loss_cls.mean().item(), t_rpn_loss_box.mean().item(), t_RCNN_loss_cls.mean().item(), t_RCNN_loss_box.mean().item()))

                s_loss_temp = 0
                s_loss_rpn_cls = 0
                s_loss_rpn_box = 0
                s_loss_rcnn_cls = 0
                s_loss_rcnn_box = 0
                start = time.time()
        
        
        if epoch % 1 == 0:
            save_name = os.path.join(output_dir,
                                 'epoch_{}.pth'.format(
                                     epoch,))
            test_output_dir = get_output_dir(imdb_tv, save_name.split('/')[-1])
            print(test_output_dir)
            with torch.no_grad():
                aps = test_scl(fasterRCNN, test_output_dir, dataset_tv, cfg, args.class_agnostic, imdb_tv, args.dataset_t)
            s = ''
            for ap in aps:
                s += str(ap) + ' ' 
            s += str(np.mean(aps))
            mAP = np.mean(aps)
            is_best = mAP > best_score
            best_score = max(best_score, mAP)
            save_checkpoint({
            'session': args.session,
            'epoch': epoch,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            'best_score': mAP
            }, save_name, is_best, best_score)
            print('save model: {}'.format(save_name))
            fasterRCNN.train()