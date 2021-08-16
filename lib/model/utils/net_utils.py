# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,Function
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_overlaps
from model.roi_layers import nms
#from model.roi_crop.functions.roi_crop import RoICropFunction
import cv2
import pdb
import random
from torch.utils.data.sampler import Sampler
from sklearn.cluster import KMeans
import shutil
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rc('font',family='Times New Roman',size=12)
from sklearn.manifold import TSNE
def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else np.array(x)

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


class EFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(EFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        # inputs = F.sigmoid(inputs)
        P = torch.softmax(inputs, dim=-1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * torch.exp(-self.gamma * probs) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = torch.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = torch.softmax(inputs, dim=-1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)


            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class FocalPseudo(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,threshold=0.8):
        super(FocalPseudo, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)*1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.threshold = threshold

    def forward(self, inputs):
        N = inputs.size(0)
        C = inputs.size(1)
        inputs = inputs[0,:,:]
        #print(inputs)
        #pdb.set_trace()
        inputs,ind = torch.max(inputs,1)
        ones = torch.ones(inputs.size()).cuda()
        value = torch.where(inputs>self.threshold,inputs,ones)
        #
        #pdb.set_trace()
        #ind
        #print(value)
        try:
            ind = value.ne(1)
            indexes = torch.nonzero(ind)
            #value2 = inputs[indexes]
            inputs = inputs[indexes]
            log_p = inputs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        except:
            #inputs = inputs#[indexes]
            log_p = value.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        #batch_loss = batch_loss #* weight
        if self.size_average:
            try:
                loss = batch_loss.mean() #+ 0.1*balance
            except:
                pdb.set_trace()
        else:
            loss = batch_loss.sum()
        return loss
def CrossEntropy(output, label):
        criteria = torch.nn.CrossEntropyLoss()
        loss = criteria(output, label)
        return loss
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()

    norm = (clip_norm / max(totalnorm, clip_norm))
    #print(norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 204), 3)
            # cv2.rectangle(im, (bbox[0], bbox[1] - 45), (bbox[0]+250, bbox[1] + 5), (255, 0, 0), thickness=-1)
            cv2.putText(im, '%s: %.2f' % (class_name, score), (bbox[0], bbox[1] - 6), cv2.FONT_HERSHEY_PLAIN,
                        2, (100, 100, 100), thickness=3)
        # if score > thresh:
        #     cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        #     cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #                 1.0, (0, 0, 255), thickness=1)
    return im

def show_results(img, image_file, results, num_classes, threshold=0.6, save_fig=True):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, 4]
        if threshold and score < threshold:
            continue

        label = int(results[i, 5])
        iou_before = results[i, 6]
        iou_after = results[i, 7]
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        display_text = '%.2f, %.2f, %.2f' % (score,iou_before, iou_after)
        ax.text(xmin, ymin, display_text, bbox={'facecolor':color, 'alpha':0.5})
    if save_fig:
        plt.savefig(image_file, bbox_inches="tight")
    #plt.show()

def show_results_score(img, image_file, results, num_classes, threshold=0.6, save_fig=True):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, 4]
        #norm = results[i, -1]
        if threshold and score < threshold:
            continue

        label = int(results[i, 5])
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        #display_text = '%.2f %.2f' % (score, norm)
        ax.text(xmin, ymin, '', bbox={'facecolor':color, 'alpha':0.5})
    if save_fig:
        plt.savefig(image_file, bbox_inches="tight")
    #plt.show()

def show_rois(img, image_file, results, save_fig=True):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, 2)).tolist()

    for i in range(0, results.shape[0]):
        color = colors[0]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
    if save_fig:
        plt.savefig(image_file, bbox_inches="tight")
        
def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
import math
def calc_supp(iter,iter_total=80000):
    p = float(iter) / iter_total
    #print(math.exp(-10*p))
    return 2 / (1 + math.exp(-10*p)) - 1
# def adjust_learning_rate(optimizer, decay=0.1,lr_init = 0.001):
#     """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay * lr_init#param_group['lr']



def save_checkpoint(state, filename, is_best, best_score=0):
    output_dir = os.path.dirname(filename)
    new_filename = os.path.join(output_dir, 'checkpoint.pth')
    torch.save(state, new_filename)
    if is_best:
        output_dir = os.path.dirname(filename)
        filename_best = os.path.join(output_dir, 'model_best'+str(best_score)+'.pth')
        shutil.copyfile(new_filename, filename_best)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,\
     sigma=1.0, dim=[1], reduction='mean'):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    if reduction == 'mean':
        loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta
'''
def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2 # random.randint(1, 8)
    H = 5 # random.randint(1, 8)
    W = 4 # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()
   
    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)    
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]


    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:,:,:,1], grid_clone.data[:,:,:,0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()
'''

def proposals_to_centers(proposals):
    """
    :param proposals: [N, 5], (b_ix, x1, y1, x2, y2)
    :return: centers [N, 2], (b_ix, center_x, center_y)
    """
    cx = (proposals[:, 3] + proposals[:, 1]) / 2.0
    cy = (proposals[:, 4] + proposals[:, 2]) / 2.0
    center = np.vstack([cx, cy]).transpose()
    return center

def compute_cluster_targets(proposals, features, N_cluster=4, threshold=128):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...), N = 512
        features: [N, 4096],
    Return:
        batch_rois: [N_cluster, 128, 4096]
        batch_cluster_center: [N_cluster, 2], (center_x, center_y)
    '''
    proposals_np = to_np_array(proposals)
    features_np = to_np_array(features)
    centers = proposals_to_centers(proposals_np)

    """
    KMeans part
    """
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(centers)

    cluster_center = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    batch_rois_cluster = []
    for cluster_idx in range(0, N_cluster):
        keep_ix = np.where(cluster_labels[:] == cluster_idx)[0]

        if keep_ix.shape[0] < threshold:
            keep_ix_new = np.random.choice(keep_ix.shape[0], threshold, replace=True)
            keep_ix2 = keep_ix[keep_ix_new]
            batch_rois_tmp = features_np[keep_ix2]
        else:
            keep_ix2 = keep_ix[0:threshold]
            batch_rois_tmp = features_np[keep_ix2]


        # batch_rois_tmp = features[keep_ix]
        batch_rois_cluster.append(batch_rois_tmp)

    batch_rois_cluster = np.stack(batch_rois_cluster, axis=0) # (N_cluster, threshold, 4096)
    batch_rois_cluster = np.mean(batch_rois_cluster, axis=1)


    f = lambda x: (torch.from_numpy(x)).float().cuda().contiguous()
    batch_rois_cluster = f(batch_rois_cluster)
    # batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois_cluster, cluster_center

def compute_mean_cluster_targets(proposals, features, N_cluster=4, threshold=128):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...), N = 512
        features: [N, 4096],
    Return:
        batch_rois: [N_cluster, 128, 4096]
        batch_cluster_center: [N_cluster, 2], (center_x, center_y)
    '''
    proposals_np = to_np_array(proposals)
    features_np = to_np_array(features)
    centers = proposals_to_centers(proposals_np)

    """
    KMeans part
    """
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(centers)

    cluster_center = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    batch_rois_cluster = []
    for cluster_idx in range(0, N_cluster):
        keep_ix = np.where(cluster_labels[:] == cluster_idx)[0]

        batch_rois_tmp = features_np[keep_ix].mean(0)

        # batch_rois_tmp = features[keep_ix]
        batch_rois_cluster.append(batch_rois_tmp)

    batch_rois_cluster = np.stack(batch_rois_cluster, axis=0) # (N_cluster, threshold, 4096)


    f = lambda x: (torch.from_numpy(x)).float().cuda().contiguous()
    batch_rois_cluster = f(batch_rois_cluster)
    # batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois_cluster, cluster_center

def compute_weighted_mean_cluster_targets(proposals, features, prob, N_cluster=4, threshold=128):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...), N = 512
        features: [N, 4096],
    Return:
        batch_rois: [N_cluster, 128, 4096]
        batch_cluster_center: [N_cluster, 2], (center_x, center_y)
    '''
    proposals_np = to_np_array(proposals)
    features_np = to_np_array(features)
    centers = proposals_to_centers(proposals_np)
    prob_np = to_np_array(prob)
    obj_prob_np = 1 - prob_np[:,0]
    """
    KMeans part
    """
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(centers)

    cluster_center = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    batch_rois_cluster = []
    for cluster_idx in range(0, N_cluster):
        keep_ix = np.where(cluster_labels[:] == cluster_idx)[0]

        batch_rois_tmp = features_np[keep_ix]
        batch_rois_weight = obj_prob_np[keep_ix] / obj_prob_np[keep_ix].sum()
        batch_rois_tmp = (batch_rois_tmp * batch_rois_weight.reshape(-1,1)).mean(0)

        batch_rois_cluster.append(batch_rois_tmp)

    batch_rois_cluster = np.stack(batch_rois_cluster, axis=0) # (N_cluster, threshold, 4096)


    f = lambda x: (torch.from_numpy(x)).float().cuda().contiguous()
    batch_rois_cluster = f(batch_rois_cluster)
    # batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois_cluster, cluster_center
def compute_entropy(predict, softmax=True):
    n, k = predict.size()
    if softmax:
        predict_prob = torch.softmax(predict, dim=-1)
    else:
        predict_prob = predict
    loss_ent = -1 * torch.mean(torch.sum(predict_prob *
                                             (torch.log(predict_prob + 1e-5)), 1))
    
    return loss_ent


def accuracy(output, target, topk=(1, ), ignore_index=-1):
    """Computes the precision@k for the specified values of k"""
    keep = torch.nonzero(target != ignore_index).squeeze()
    #logger.info('target.shape:{0}, keep.shape:{1}'.format(target.shape, keep.shape))
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def select_roi(all_rois, gt_boxes=None, num_boxes=None):
    overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

    max_overlaps, gt_assignment = torch.max(overlaps, 2)

    batch_size = overlaps.size(0)
    num_proposal = overlaps.size(1)
    num_boxes_per_img = overlaps.size(2)

    offset = torch.arange(0, batch_size)*gt_boxes.size(1)
    offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

    # changed indexing way for pytorch 1.0
    labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

    labels_batch = labels.new(batch_size, num_proposal).zero_()

    # Guard against the case when an image has fewer than max_fg_rois_per_image
    # foreground RoIs
    for i in range(batch_size):

        fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
        labels_batch[i][fg_inds] = labels[i][fg_inds]

    return labels_batch



def cluster_rois(batch_size, rois, cls_prob, pooled_feat):
        rois = rois[:,:,1:]
        for i in range(batch_size):
            boxes = rois[i]
            scores= cls_prob.view(batch_size, -1, self.n_classes)[i]
            feats = pooled_feat.view(batch_size, -1, 4096)[i]
            thresh = 0.01
            class_feats_list = []
            #pdb.set_trace()
            for j in range(self.n_classes):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = boxes[inds, :] 
                    j_feats = feats[inds,:]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    j_feats = j_feats[order]
                    
                    
                    try:
                        keep = nms(cls_dets[:,:4], cls_dets[:,-1], 0.3)
                    except:
                        print(cls_dets.size())
                    
                    keep = keep.view(-1).long()
                    num_clusters = keep.size(0)
                    cluster_feats_list = []
                    for k in range(num_clusters):
                        start = keep[k]
                        if k != num_clusters-1:
                            end = keep[k+1]
                        else:
                            end = cls_dets.size(0)
                        try:
                            c_cls_dets = cls_dets[start:end]
                        except:
                            print(order, start, end)
                        c_feats = j_feats[start:end]
                        cluster_feats = (c_feats * (c_cls_dets[:,-1] / c_cls_dets[:,-1].sum()).unsqueeze(1)).sum(0)
                        cluster_feats_list.append(cluster_feats)
                    class_feats = torch.stack(cluster_feats_list, 0).mean(0)
                else:
                    class_feats = (feats * (scores[:,j] / scores[:,j].sum()).unsqueeze(1)).sum(0)
                class_feats_list.append(class_feats)
            return torch.stack(class_feats_list,0)

def cluster_rois_v1(batch_size, rois, roi_feats):
    rois = rois.view(batch_size, -1, 5)[:,:,1:]
    roi_feats = roi_feats.view(batch_size, -1, roi_feats.size(-1))
    n_rois = rois.size(1)
    score = torch.arange(n_rois, 0, -1).cuda().float() / n_rois
    cluster_roi_feats = list()

    for i in range(batch_size):
        overlaps = bbox_overlaps(rois[i], rois[i])
        overlaps[overlaps<0.5] = 0
        overlaps /= overlaps.sum(1, keepdim=True)
        fused_roi_feat = torch.matmul(overlaps, roi_feats[i])
        keep = nms(rois[i], score, 0.5)
        cluster_roi_feats.append(fused_roi_feat[keep])

    return torch.cat(cluster_roi_feats, 0)

def cluster_rois_v3(batch_size, rois, roi_feats):
    rois = rois.view(batch_size, -1, 5)[:,:,1:]
    roi_feats = roi_feats.view(batch_size, -1, roi_feats.size(-1))
    n_rois = rois.size(1)
    cluster_roi_feats = list()

    for i in range(batch_size):
        overlaps = bbox_overlaps(rois[i], rois[i])
        overlaps[overlaps<0.5] = 0
        overlaps /= overlaps.sum(1, keepdim=True)
        fused_roi_feat = torch.matmul(overlaps, roi_feats[i])
        cluster_roi_feats.append(fused_roi_feat)

    return torch.cat(cluster_roi_feats, 0)

def cluster_rois_v2(batch_size, rois, roi_feats):
    rois = rois.view(batch_size, -1, 5)[:,:,1:]
    roi_feats = roi_feats.view(batch_size, -1, roi_feats.size(-1))
    n_rois = rois.size(1)
    cluster_roi_feats = list()
    for i in range(batch_size):
        overlaps = bbox_overlaps(rois[i], rois[i])
        overlaps[overlaps<0.5] = 0
        overlaps /= overlaps.sum(1, keepdim=True)
        fused_roi_feat = torch.matmul(overlaps, roi_feats[i])
        cluster_roi_feats.append(fused_roi_feat)
    return torch.cat(cluster_roi_feats, 0)

def assign_pseudo_label(cls_prob, bg_threshold=0.75, max_n_bgs=64, fg_threshold=0.5):

    n_classes = cls_prob.size(1)
    pseudo_labels = torch.zeros(cls_prob.size(0)).cuda().long() - 1
    n_fgs = 0
    for j in range(1,n_classes):
        inds = torch.nonzero(cls_prob[:,j] >= fg_threshold).view(-1)
        pseudo_labels[inds] = j
        n_fgs += inds.size(0)
    fg_inds = (cls_prob[:,0] >= bg_threshold).nonzero().view(-1)
    if fg_inds.size(0) > max_n_bgs:
        pseudo_labels[fg_inds[:max_n_bgs]] = 0
    return pseudo_labels

def assign_real_label(all_rois, gt_boxes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # changed indexing way for pytorch 1.0
        labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

        labels_batch = labels.new(batch_size, num_proposal).zero_() -1 

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            # Select sampled values from various arrays:
            labels_batch[i][fg_inds] = labels[i][fg_inds]
            labels_batch[i][bg_inds] = 0
        return labels_batch
'''
def vis_roi_feat(dataset_name, i, s_boxes, s_gt_boxes, s_roi_feat, t_boxes, t_gt_boxes, t_roi_feat,version='default'):
    os.makedirs('./tsne_images/'+str(dataset_name)+'/'+version, exist_ok=True)
    path = './tsne_images/'+str(dataset_name)+'/'+version+'/'+'vis.jpg'

    iou_before = bbox_overlaps(s_boxes.squeeze(), s_gt_boxes.view(-1,5)[:,:4])
    max_iou_before = iou_before.max(1, keepdim=True)[0]
    s_pos_inds = (max_iou_before >= 0.5).nonzero().view(-1).cpu().numpy()
    s_neg_inds = (max_iou_before < 0.4).nonzero().view(-1).cpu().numpy()
    n_s = s_roi_feat.size(0)
    
    iou_before = bbox_overlaps(t_boxes.squeeze(), t_gt_boxes.view(-1,5)[:,:4])
    max_iou_before = iou_before.max(1, keepdim=True)[0]
    t_pos_inds = (max_iou_before >= 0.5).nonzero().view(-1).cpu().numpy() + n_s
    t_neg_inds = (max_iou_before < 0.4).nonzero().view(-1).cpu().numpy() + n_s

    roi_feat = torch.cat([s_roi_feat, t_roi_feat], 0)
    roi_embed = TSNE(n_components=2).fit_transform(roi_feat.detach().cpu().numpy())
    
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(roi_embed[s_pos_inds, 0], roi_embed[s_pos_inds, 1], c='green')
    ax.scatter(roi_embed[s_neg_inds, 0], roi_embed[s_neg_inds, 1], c='blue')
    ax.scatter(roi_embed[t_pos_inds, 0], roi_embed[t_pos_inds, 1], c='cyan')
    ax.scatter(roi_embed[t_neg_inds, 0], roi_embed[t_neg_inds, 1], c='red')
    plt.savefig(path)
    print('saved  '+path)
    plt.close()
'''
def vis_roi_feat(dataset_name, step, s_all_roi_feat, s_all_overlaps, t_all_roi_feat, t_all_overlaps, version='default'):
    os.makedirs('./tsne_images/'+str(dataset_name)+'/'+version, exist_ok=True)
    path = './tsne_images/'+str(dataset_name)+'/'+version+'/'+ str(step) + '.jpg'

    s_pos_inds = (s_all_overlaps >= 0.5).nonzero().view(-1).cpu().numpy()
    s_neg_inds = (s_all_overlaps < 0.4).nonzero().view(-1).cpu().numpy()
    n_s = s_all_roi_feat.size(0)
    
    t_pos_inds = (t_all_overlaps >= 0.5).nonzero().view(-1).cpu().numpy() + n_s
    t_neg_inds = (t_all_overlaps < 0.4).nonzero().view(-1).cpu().numpy() + n_s

    roi_feat = torch.cat([s_all_roi_feat, t_all_roi_feat], 0)
    roi_embed = TSNE(n_components=2, perplexity=15, learning_rate=400).fit_transform(roi_feat.detach().cpu().numpy())
    
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(roi_embed[s_pos_inds, 0], roi_embed[s_pos_inds, 1], c='green')
    ax.scatter(roi_embed[s_neg_inds, 0], roi_embed[s_neg_inds, 1], c='blue')
    ax.scatter(roi_embed[t_pos_inds, 0], roi_embed[t_pos_inds, 1], c='cyan')
    ax.scatter(roi_embed[t_neg_inds, 0], roi_embed[t_neg_inds, 1], c='red')
    plt.savefig(path)
    print('saved  '+path)
    plt.close()