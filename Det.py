import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utils,imgproc,file_utils
import json
import zipfile

from .craft import CRAFT

from collections import OrderedDict

class Detection():
    def __init__(self, model_path):
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.canvas_size = 1280
        self.mag_ratio = 1.5 
        self.net = CRAFT()
        self.cuda = torch.cuda.is_available()

        print('Loading weights from checkpoint (' + model_path + ')')
        if self.cuda:
            self.net.load_state_dict(self._copyStateDict(torch.load(model_path)))
        else:
            self.net.load_state_dict(self._copyStateDict(torch.load(model_path, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()
    def _copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    def detect(self,image):
        # initialize
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()


        # Post-processing
        boxes, _, _ = craft_utils.getDetBoxes_core(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        return boxes