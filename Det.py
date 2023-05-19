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
    def __init__(self, model_path, show_time):
        self.model_path = model_path
        self.show_time = show_time
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
        text_threshold = 0.7
        link_threshold = 0.4
        low_text = 0.4
        canvas_size = 1280
        mag_ratio = 1.5 
        net = CRAFT()
        cuda = torch.cuda.is_available()

        print('Loading weights from checkpoint (' + self.model_path + ')')
        if cuda:
            net.load_state_dict(self._copyStateDict(torch.load(self.model_path)))
        else:
            net.load_state_dict(self._copyStateDict(torch.load(self.model_path, map_location='cpu')))

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        t0 = time.time()
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, _, _ = craft_utils.getDetBoxes_core(score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        t1 = time.time() - t1

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes