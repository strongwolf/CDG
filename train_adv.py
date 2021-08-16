import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, FocalLoss, sampler, EFocalLoss

from model.utils.parser_func_multi import parse_args, set_dataset_args
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
    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

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
    
    output_dir = args.save_dir + "/"  + args.net + "/" + args.dataset + '2' + args.dataset_t +'/' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.cuda:
        cfg.CUDA = True

    from model.faster_rcnn.resnet_adv import resnet

    if args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,lc=args.lc, gc=args.gc)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)

    else:
        print("network is not defined")
        pdb.set_trace()

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
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch =  int(len(dataloader_s) / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    count_iter = 0
    counter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()
        if epoch-1 in args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
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
            #eta = 1.0
            count_iter += 1
            if args.cuda:
                im_data = data_s[0].cuda()
                im_info = data_s[1].cuda()
                gt_boxes = data_s[2].cuda()
                num_boxes = data_s[3].cuda()
            # print(im_data.shape)
            if(len(im_data.size()) != 4):
                print("skipping due to image size")
                counter += 1
                continue
            fasterRCNN.zero_grad()
            outputs = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            rois, cls_prob, bbox_pred = outputs['predict']
            rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox = outputs['loss']
            out_d_pixel, out_d = outputs['d_loss']
            rois_label = outputs['rois_label'] 
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # domain label
            domain_s = torch.zeros(out_d.size(0)).long().cuda()
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            if args.cuda:
                im_data = data_t[0].cuda()
                im_info = data_t[1].cuda()
                gt_boxes = data_t[2].cuda()
                num_boxes = data_t[3].cuda()

            # print(im_data.size())
            if(len(im_data.size()) != 4):
                print(im_data.size())  
                counter += 1
                continue
            outputs = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            out_d_pixel, out_d = outputs
            # domain label
            domain_t = torch.ones(out_d.size(0)).long().cuda()
            dloss_t = 0.5 * FL(out_d, domain_t)
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
            if args.dataset == 'sim10k':
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p) * args.eta
            else:
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p) * args.eta
            optimizer.zero_grad()
            loss.backward()
            if 'vgg' in args.net:
                nn.utils.clip_grad_norm_(fasterRCNN.parameters(), 7)
            else:
                nn.utils.clip_grad_norm_(fasterRCNN.parameters(), 10)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f dloss s pixel: %.4f dloss t pixel: %.4f eta: %.4f counter: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s, dloss_t, dloss_s_p, dloss_t_p,
                       args.eta, counter))
                counter = 0

                loss_temp = 0
                start = time.time()
        save_name = os.path.join(output_dir,
                                 'session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.lc, args.gc, args.gamma,
                                     args.session, epoch,
                                     step))
        
        if epoch % 1 == 0:
            save_name = os.path.join(output_dir,
                                 'epoch_{}.pth'.format(
                                     epoch,))
            save_checkpoint({
            'session': args.session,
            'epoch': epoch,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            'best_score': 0
            }, save_name, False, best_score)
            print('save model: {}'.format(save_name))

