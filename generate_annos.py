import os
import sys
import numpy as np
import pprint
import time
import _init_paths

import torch
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, bbox_overlaps
from model.utils.parser_func_multi import parse_args,set_dataset_args
import pdb
from lxml import etree, objectify
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
np.set_printoptions(suppress=True)
import shutil
mpl.use('agg')
try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3

def test_scl(fasterRCNN, output_dir, dataset, cfg, class_agnostic, imdb, dataset_name, version='default'):

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
							shuffle=False, num_workers=0,
							pin_memory=True)
	data_iter = iter(dataloader)
	num_images = len(imdb.image_index)
	fasterRCNN.eval()
	wbg = True
	dict_num = 192
	theta = 0.01
	lamda = 0.1
	dict_root_dir = ''
	if wbg:
		dicts_file = os.path.join(dict_root_dir, 'wbg_learn_dicts_' + str(dict_num) + '_' + str(theta) + '.pkl')
		inv_file = os.path.join(dict_root_dir, 'wbg_learn_inv_' + str(dict_num) + '_'+ str(theta) + '_' + str(lamda) + '.pkl')
	else:
		dicts_file = os.path.join(dict_root_dir, 'gt_learn_dicts_' + str(dict_num) + '_' + str(theta) + '.pkl')
		inv_file = os.path.join(dict_root_dir, 'learn_inv_' + str(dict_num) + '_' + str(theta) + '_' + str(lamda) + '.pkl')
	save = True
	stat = True
	if save:
		if 'sim10k' in dataset_name:
			anno_dir = os.path.join(imdb._data_path, 'annos_pseudo_sim10k_'+version)
			ori_anno_dir = os.path.join(imdb._data_path, 'annos')
			thresh_cls_list = [0, 0.]
			abs_r_list = [0, 0.1]
		if 'city' in dataset_name:
			anno_dir = os.path.join(imdb._data_path, 'annos_pseudo_city_'+version)
			ori_anno_dir = os.path.join(imdb._data_path, 'annos')
			thresh_cls_list = [0, 0.,0.,0.,0.,0.,0.,0.,0.]
			abs_r_list = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
		if 'water' in dataset_name:
			anno_dir = os.path.join(imdb._data_path, 'Annotations_pseudo_watercolor_'+version)
			ori_anno_dir = os.path.join(imdb._data_path, 'Annotations')
			thresh_cls_list = [0, 0.,0.,0.,0.,0.,0.]
			abs_r_list = [0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]
		if os.path.exists(anno_dir):
			shutil.rmtree(anno_dir)
		
		shutil.copytree(ori_anno_dir, anno_dir)
	
	with open(inv_file, 'rb') as f:
		inv_matrix = torch.from_numpy(pickle.load(f)).cuda().float()
	with open(dicts_file, 'rb') as f:
		dicts = pickle.load(f)
		if type(dicts) == list:
			num_classes = len(dicts)
			dict_num = dicts[0].shape[1]
			dicts  = np.stack(dicts, 1).reshape(-1, num_classes*dict_num)
		dicts_matrix = torch.from_numpy(dicts).cuda().float()
		num_classes = int(dicts_matrix.size(1) / dict_num)
	num_pseudo_box = 0
	pseudo_res_list = []
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
			#print(num_boxes, imdb.image_path_at(i))
			det_tic = time.time()
			with torch.no_grad():
				outputs = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
			rois_1, cls_prob_1, bbox_pred_1 = outputs['predict']
			base_feat = outputs['base_feat']
			scores = cls_prob_1.data
			boxes_1 = rois_1.data[:, :, 1:5]

			if cfg.TEST.BBOX_REG:
				# Apply bounding-box regression deltas
				box_deltas = bbox_pred_1.data
				if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
					if class_agnostic:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
								+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						box_deltas = box_deltas.view(1, -1, 4)
					else:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
								+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

				pred_boxes = bbox_transform_inv(boxes_1, box_deltas, 1)
				pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
			
			else:
				# Simply repeat the boxes, once for each class
				pred_boxes = np.tile(boxes_1, (1, scores.shape[1]))
			
			pred_boxes = pred_boxes.squeeze().view(-1,4)
			pred_rois = torch.cat([torch.zeros(pred_boxes.size(0)).cuda().view(-1,1), pred_boxes[:,:4]], 1)
			pred_feat = fasterRCNN.RCNN_roi_align(base_feat, pred_rois.view(-1, 5))
			pred_feat = fasterRCNN._head_to_tail(pred_feat)
			pred_score = fasterRCNN.RCNN_cls_score(pred_feat)
			pred_prob = torch.softmax(pred_score, 1)
			
			overlaps = bbox_overlaps(pred_boxes, gt_boxes.view(-1,5)[:,:4])
			max_overlaps, index = overlaps.max(1)
			gt_assignments = gt_boxes.view(-1,5)[index,-1]
			gt_assignments[max_overlaps==0] = 0
			pred_boxes /= data[1][0][2].item()
			scores = pred_prob.data.squeeze()
			
			det_toc = time.time()
			detect_time = det_toc - det_tic
			misc_tic = time.time()
			pred_norm = torch.norm(pred_feat, p=2, dim=1, keepdim=True)
			pred_feat = pred_feat.div(pred_norm)
			alpha_matrix = torch.matmul(inv_matrix, pred_feat.transpose(1,0).contiguous())
			res_norm_matrix_list = []
			alpha_norm_matrix_list = []
			r_norm_matrix_list = []
			for j in xrange(0, num_classes):
				j_alpha_matrix = alpha_matrix[j*dict_num:(j+1)*dict_num,:]
				j_dicts_matrix = dicts_matrix[:,j*dict_num:(j+1)*dict_num]
				res_norm_matrix = torch.norm(pred_feat.transpose(1,0).contiguous() - torch.matmul(j_dicts_matrix,j_alpha_matrix), p=2, dim=0)
				alpha_norm_matrix = torch.norm(j_alpha_matrix, p=2, dim=0)
				r_norm_matrix = res_norm_matrix / alpha_norm_matrix
				res_norm_matrix_list.append(res_norm_matrix)
				alpha_norm_matrix_list.append(alpha_norm_matrix)
				r_norm_matrix_list.append(r_norm_matrix)
			res_norm_matrix_list = torch.stack(res_norm_matrix_list, 1)
			alpha_norm_matrix_list = torch.stack(alpha_norm_matrix_list, 1)
			r_norm_matrix_list = torch.stack(r_norm_matrix_list, 1)
			sorted_res, sorted_res_index = torch.sort(res_norm_matrix_list, 1)
			pseudo_boxes = []
			for j in xrange(1, imdb.num_classes):
				thresh = thresh_cls_list[j]
				cls_scores = scores[:,j]
				inds = torch.nonzero(cls_scores>thresh).view(-1)
				cls_scores = cls_scores[inds]
				if class_agnostic:
					cls_boxes = pred_boxes[inds]
				else:
					cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
				cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
				_, order = torch.sort(cls_scores, 0, True)
				cls_scores = cls_scores[order]
				cls_dets = cls_dets[order]
				threshhold = 0.4
				keep = nms(cls_dets[:,:4], cls_dets[:,-1], threshhold)
				keep = keep.view(-1).long()
				inds = inds[order][keep]
    
				j_sorted_res_index = sorted_res_index[inds]
				j_sorted_res = sorted_res[inds]
				
				if wbg:
					index_1 = j_sorted_res_index[:,0] == j
				else:
					index_1 = j_sorted_res_index[:,0] == j-1
				index_2 = (j_sorted_res[:,1] - j_sorted_res[:,0]) >= abs_r_list[j]
				pseudo_ids = (index_1 & index_2).nonzero().view(-1)

				if pseudo_ids.numel() != 0:
					pseudo_dets = torch.cat([cls_dets[[pseudo_ids]], torch.zeros(pseudo_ids.size(0),1).cuda()+j,\
								j_sorted_res[pseudo_ids,0].view(-1,1)],1)
					pseudo_boxes.append(pseudo_dets)
					num_pseudo_box += pseudo_dets.size(0)
					pseudo_res_list.append(j_sorted_res[pseudo_ids,0])

			if len(pseudo_boxes) != 0:
				pseudo_boxes = torch.cat(pseudo_boxes, 0).cpu().numpy()
			else:
				pseudo_boxes = np.empty(0)
			misc_toc = time.time()
			nms_time = misc_toc - misc_tic

			sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
				.format(i + 1, num_images, detect_time, nms_time))
			sys.stdout.flush()
			if save:
				anno_file = os.path.join(anno_dir, imdb._image_index[i]+'.xml')
				print(anno_file)
				img = Image.open(imdb.image_path_at(i))
				w,h = img.size
				E = objectify.ElementMaker(annotate=False)
				anno_tree = E.annotation(
							E.size(
								E.width(w),
								E.height(h),
								E.depth(3),
							)
				)
				if pseudo_boxes.shape[0] != 0:
					print(pseudo_boxes.shape[0]<num_boxes.item(), ' ')
					for k in range(pseudo_boxes.shape[0]):
						box = pseudo_boxes[k]
						x1 = max(1, int(box[0])+1)
						y1 = max(1, int(box[1])+1)
						x2 = min(w, int(box[2])+1)
						y2 = min(h, int(box[3])+1)
						label = imdb._classes[int(box[-2])]
						norm = box[-1]
						box_tree = E.object(
									E.name(label),
									E.truncated(0),
									E.difficult(0),
									E.norm(norm),
									E.bndbox(
										E.xmin(x1),
										E.ymin(y1),
										E.xmax(x2),
										E.ymax(y2),
									)
						)
						anno_tree.append(box_tree)
				etree.ElementTree(anno_tree).write(anno_file, pretty_print=True)

if __name__ == '__main__':

	args = parse_args()

	print('Called with args:')
	print(args)
	print(args.vis, "   ", args.dataset)
	args = set_dataset_args(args,test=False)
	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	np.random.seed(cfg.RNG_SEED)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)


	cfg.TRAIN.USE_FLIPPED = False
	imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name_target, False)
	imdb.competition_mode(on=True)

	print('{:d} roidb entries'.format(len(roidb)))

	from model.faster_rcnn.vgg16 import vgg16
	from model.faster_rcnn.resnet_adv import resnet

	if args.net == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res50':
		fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,
							lc=args.lc, gc=args.gc)
	elif args.net == 'res101':
		fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
							lc=args.lc, gc=args.gc)
	else:
		print("network is not defined")
		pdb.set_trace()

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
	test_scl(fasterRCNN, output_dir, dataset, cfg, args.class_agnostic, imdb, args.dataset)






