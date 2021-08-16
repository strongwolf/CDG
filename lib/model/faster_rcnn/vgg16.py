import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_vgg import _fasterRCNN
from model.utils.config import cfg

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = cfg.VGG_PATH
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:5])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[5:10])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[10:17])
    self.RCNN_base4 = nn.Sequential(*list(vgg.features._modules.values())[17:24])
    self.RCNN_base5 = nn.Sequential(*list(vgg.features._modules.values())[24:-1])
    feat_d = 4096

    #for layer in range(10):
      #for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    self.RCNN_top = vgg.classifier

    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

