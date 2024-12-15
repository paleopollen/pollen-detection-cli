#  Copyright 2024 The Board of Trustees of the University of Illinois. All Rights Reserved.
#
#  Licensed under the terms of Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  The License is included in the distribution as LICENSE file.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os.path as path
import pickle

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform
from torch.utils.data import Dataset
from torchvision import transforms


class PollenDet4Eval(Dataset):
    def __init__(self, path_to_image='',
                 path_to_annot='',
                 dbinfo=None,
                 size=[512, 512],
                 set_name='train'):

        self.path_to_image = path_to_image
        self.path_to_annot = path_to_annot
        self.transform = transform
        self.dbinfo = dbinfo
        if set_name == 'val':
            set_name = 'test'
        self.set_name = set_name
        self.size = size
        self.resizeFactor = size[0] / 1000

        self.sampleList = self.dbinfo[set_name + '_det_list']
        self.clsNameList = self.dbinfo[set_name + '_det_className']
        self.clsIDList = self.dbinfo[set_name + '_det_classID']

        self.TFNormalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.current_set_len = len(self.sampleList)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFresize = transforms.Resize((self.size[0], self.size[1]))

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curClassName = self.dbinfo['train_det_className'][idx]
        curImageName = path.join(self.path_to_image, self.dbinfo['train_det_list'][idx] + '.png')
        curPickleName = path.join(self.path_to_annot, self.dbinfo['train_det_list'][idx] + '.pkl')

        with open(curPickleName, 'rb') as handle:
            annot = pickle.load(handle)

        image = PIL.Image.open(curImageName)

        if self.set_name == 'train' and np.random.random(1) > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for i in range(annot['coord_peaks'].shape[0]):
                y, x = annot['coord_peaks'][i]  # annot['size'][0]:y, annot['size'][1]:x
                # ynew = y
                # xnew = annot['size'][1]-x
                annot['coord_peaks'][i][1] = annot['size'][1] - x

        image = np.array(image)

        label = np.zeros((annot['size'][0], annot['size'][1]), dtype=np.float32)
        mask_distanceTransform = label * 0.
        mask_peaks = label * 0.
        mask_radius = label * 0.
        mask_voteX = label * 0.
        mask_voteY = label * 0.
        mask_overlap = label * 0.

        labelOrgSize = np.zeros((annot['size'][0], annot['size'][1]), dtype=np.float32)
        mask_peaksOrgSize = labelOrgSize * 0.

        for i in range(annot['coord_peaks'].shape[0]):
            y, x = annot['coord_peaks'][i]
            r = annot['mask_radius'][i]
            mask_peaks[y - 10:y + 10, x - 10:x + 10] = 1
            mask_radius[y - 10:y + 10, x - 10:x + 10] = r
            # mask_peaks[y, x] = 1
            # mask_radius[y, x] = r

            mask_peaksOrgSize[y, x] = 1

            mask_x, mask_y = np.asarray(range(label.shape[1])).astype(np.float), np.asarray(
                range(label.shape[0])).astype(np.float)
            mask_x, mask_y = np.meshgrid(mask_x, mask_y)
            mask_x = float(x) - mask_x
            mask_y = float(y) - mask_y

            tmpDistTransform = np.sqrt(mask_x * mask_x + mask_y * mask_y)
            tmpmask_vote = tmpDistTransform <= r
            label[tmpmask_vote] = 1.0
            labelOrgSize[tmpmask_vote] = 1
            mask_voteX[tmpmask_vote] = mask_x[tmpmask_vote]
            mask_voteY[tmpmask_vote] = mask_y[tmpmask_vote]
            mask_distanceTransform[tmpmask_vote] = r - tmpDistTransform[tmpmask_vote]
            mask_overlap += tmpmask_vote.astype(np.float)

        mask_distanceTransformOrgSize = mask_distanceTransform.copy().astype(np.float32)

        mask_overlap = mask_overlap > 1
        mask_overlap = mask_overlap.astype(np.float32)
        # mask_overlap
        mask_voteX = mask_voteX.astype(np.float32) / 100.0 / self.resizeFactor
        mask_voteY = mask_voteY.astype(np.float32) / 100.0 / self.resizeFactor
        mask_distanceTransform = mask_distanceTransform.astype(
            np.float32) / 100.0 / self.resizeFactor  # factor=size[0]/1000
        mask_peaks = mask_peaks.astype(np.float32)
        mask_peaksOrgSize = mask_peaksOrgSize.astype(np.float32)
        mask_radius = mask_radius.astype(np.float32)
        mask_radiusOrgSize = mask_radius.copy()
        mask_radius = mask_radius / 100.0 / self.resizeFactor

        image = self.TF2tensor(image)
        label = torch.from_numpy(label).unsqueeze(0)  # self.TF2tensor(label)
        labelOrgSize = torch.from_numpy(labelOrgSize).unsqueeze(0)
        mask_overlap = torch.from_numpy(mask_overlap).unsqueeze(0)  # self.TF2tensor(mask_overlap)
        mask_voteX = torch.from_numpy(mask_voteX).unsqueeze(0)  # self.TF2tensor(mask_voteX)
        mask_voteY = torch.from_numpy(mask_voteY).unsqueeze(0)  # self.TF2tensor(mask_voteY)
        mask_distanceTransform = torch.from_numpy(mask_distanceTransform).unsqueeze(
            0)  # self.TF2tensor(mask_distanceTransform)
        mask_peaks = torch.from_numpy(mask_peaks).unsqueeze(0)  # self.TF2tensor(mask_peaks)
        mask_peaksOrgSize = torch.from_numpy(mask_peaksOrgSize).unsqueeze(0)
        mask_radius = torch.from_numpy(mask_radius).unsqueeze(0)  # self.TF2tensor(mask_radius)
        mask_distanceTransformOrgSize = torch.from_numpy(mask_distanceTransformOrgSize).unsqueeze(0)

        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        mask_distanceTransform = mask_distanceTransform.unsqueeze(0)
        mask_overlap = mask_overlap.unsqueeze(0)
        mask_voteX = mask_voteX.unsqueeze(0)
        mask_voteY = mask_voteY.unsqueeze(0)
        mask_peaks = mask_peaks.unsqueeze(0)
        mask_radius = mask_radius.unsqueeze(0)

        image = F.interpolate(image, size=(self.size[0], self.size[1]), mode='bilinear', align_corners=True)
        label = F.interpolate(label, size=(self.size[0], self.size[1]), mode='nearest')
        mask_distanceTransform = F.interpolate(mask_distanceTransform, size=(self.size[0], self.size[1]),
                                               mode='nearest')
        mask_overlap = F.interpolate(mask_overlap, size=(self.size[0], self.size[1]), mode='nearest')
        mask_voteX = F.interpolate(mask_voteX, size=(self.size[0], self.size[1]), mode='nearest')
        mask_voteY = F.interpolate(mask_voteY, size=(self.size[0], self.size[1]), mode='nearest')
        mask_peaks = F.interpolate(mask_peaks, size=(self.size[0], self.size[1]), mode='nearest')
        mask_radius = F.interpolate(mask_radius, size=(self.size[0], self.size[1]), mode='nearest')

        image = image.squeeze(0)
        label = label.squeeze(0)
        mask_distanceTransform = mask_distanceTransform.squeeze(0)
        mask_overlap = mask_overlap.squeeze(0)
        mask_voteX = mask_voteX.squeeze(0)
        mask_voteY = mask_voteY.squeeze(0)
        mask_peaks = mask_peaks.squeeze(0)
        mask_radius = mask_radius.squeeze(0)
        image = self.TFNormalize(image)

        return image, label, mask_distanceTransform, mask_overlap, mask_voteX, mask_voteY, mask_peaks, mask_radius, labelOrgSize, mask_peaksOrgSize, mask_distanceTransformOrgSize, mask_radiusOrgSize


class PollenDet(Dataset):
    def __init__(self, path_to_image='',
                 path_to_annot='',
                 dbinfo=None,
                 size=[512, 512],
                 set_name='train'):

        self.path_to_image = path_to_image
        self.path_to_annot = path_to_annot
        self.transform = transform
        self.dbinfo = dbinfo
        if set_name == 'val':
            set_name = 'test'
        self.set_name = set_name
        self.size = size
        self.resizeFactor = size[0] / 1000

        self.sampleList = self.dbinfo[set_name + '_det_list']
        self.clsNameList = self.dbinfo[set_name + '_det_className']
        self.clsIDList = self.dbinfo[set_name + '_det_classID']

        self.TFNormalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.current_set_len = len(self.sampleList)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFresize = transforms.Resize((self.size[0], self.size[1]))

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curClassName = self.dbinfo['train_det_className'][idx]
        curImageName = path.join(self.path_to_image, self.dbinfo['train_det_list'][idx] + '.png')
        curPickleName = path.join(self.path_to_annot, self.dbinfo['train_det_list'][idx] + '.pkl')

        with open(curPickleName, 'rb') as handle:
            annot = pickle.load(handle)

        image = PIL.Image.open(curImageName)

        if self.set_name == 'train' and np.random.random(1) > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for i in range(annot['coord_peaks'].shape[0]):
                y, x = annot['coord_peaks'][i]  # annot['size'][0]:y, annot['size'][1]:x
                # ynew = y
                # xnew = annot['size'][1]-x
                annot['coord_peaks'][i][1] = annot['size'][1] - x

        image = np.array(image)
        times = np.random.randint(1)
        if times != 0:
            if times == 1:
                for i in range(annot['coord_peaks'].shape[0]):
                    y, x = annot['coord_peaks'][i]
                    annot['coord_peaks'][i][0] = annot['coord_peaks'].shape[1] - x
                    annot['coord_peaks'][i][1] = y
                    annot['size'] = (annot['size'][1], annot['size'][0])
            elif times == 2:
                for i in range(annot['coord_peaks'].shape[0]):
                    y, x = annot['coord_peaks'][i]
                    annot['coord_peaks'][i][0] = annot['coord_peaks'].shape[0] - y
                    annot['coord_peaks'][i][1] = annot['coord_peaks'].shape[1] - x
            elif times == 3:
                for i in range(annot['coord_peaks'].shape[0]):
                    y, x = annot['coord_peaks'][i]
                    annot['coord_peaks'][i][0] = x
                    annot['coord_peaks'][i][1] = annot['coord_peaks'].shape[0] - y
                    annot['size'] = (annot['size'][1], annot['size'][0])
            # for _ in range(times):
            image = np.rot90(image, times).copy()

        label = np.zeros((annot['size'][0], annot['size'][1]), dtype=np.float32)
        mask_distanceTransform = label * 0.
        mask_peaks = label * 0.
        mask_radius = label * 0.
        mask_voteX = label * 0.
        mask_voteY = label * 0.
        mask_overlap = label * 0.

        for i in range(annot['coord_peaks'].shape[0]):
            y, x = annot['coord_peaks'][i]
            r = annot['mask_radius'][i]
            mask_peaks[y - 10:y + 10, x - 10:x + 10] = 1
            mask_radius[y - 10:y + 10, x - 10:x + 10] = r
            # mask_peaks[y, x] = 1
            # mask_radius[y, x] = r

            mask_x, mask_y = np.asarray(range(label.shape[1])).astype(np.float), np.asarray(
                range(label.shape[0])).astype(np.float)
            mask_x, mask_y = np.meshgrid(mask_x, mask_y)
            mask_x = float(x) - mask_x
            mask_y = float(y) - mask_y

            tmpDistTransform = np.sqrt(mask_x * mask_x + mask_y * mask_y)
            tmpmask_vote = tmpDistTransform <= r
            label[tmpmask_vote] = 1.0
            mask_voteX[tmpmask_vote] = mask_x[tmpmask_vote]
            mask_voteY[tmpmask_vote] = mask_y[tmpmask_vote]
            mask_distanceTransform[tmpmask_vote] = r - tmpDistTransform[tmpmask_vote]
            mask_overlap += tmpmask_vote.astype(np.float)

        mask_overlap = mask_overlap > 1
        mask_overlap = mask_overlap.astype(np.float32)
        # mask_overlap
        mask_voteX = mask_voteX.astype(np.float32) / 100.0 / self.resizeFactor
        mask_voteY = mask_voteY.astype(np.float32) / 100.0 / self.resizeFactor
        mask_distanceTransform = mask_distanceTransform.astype(
            np.float32) / 100.0 / self.resizeFactor  # factor=size[0]/1000
        mask_peaks = mask_peaks.astype(np.float32)
        mask_radius = mask_radius.astype(np.float32) / 100.0 / self.resizeFactor

        image = self.TF2tensor(image)
        label = torch.from_numpy(label).unsqueeze(0)  # self.TF2tensor(label)
        mask_overlap = torch.from_numpy(mask_overlap).unsqueeze(0)  # self.TF2tensor(mask_overlap)
        mask_voteX = torch.from_numpy(mask_voteX).unsqueeze(0)  # self.TF2tensor(mask_voteX)
        mask_voteY = torch.from_numpy(mask_voteY).unsqueeze(0)  # self.TF2tensor(mask_voteY)
        mask_distanceTransform = torch.from_numpy(mask_distanceTransform).unsqueeze(
            0)  # self.TF2tensor(mask_distanceTransform)
        mask_peaks = torch.from_numpy(mask_peaks).unsqueeze(0)  # self.TF2tensor(mask_peaks)
        mask_radius = torch.from_numpy(mask_radius).unsqueeze(0)  # self.TF2tensor(mask_radius)

        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        mask_distanceTransform = mask_distanceTransform.unsqueeze(0)
        mask_overlap = mask_overlap.unsqueeze(0)
        mask_voteX = mask_voteX.unsqueeze(0)
        mask_voteY = mask_voteY.unsqueeze(0)
        mask_peaks = mask_peaks.unsqueeze(0)
        mask_radius = mask_radius.unsqueeze(0)

        image = F.interpolate(image, size=(self.size[0], self.size[1]), mode='bilinear', align_corners=True)
        label = F.interpolate(label, size=(self.size[0], self.size[1]), mode='nearest')
        mask_distanceTransform = F.interpolate(mask_distanceTransform, size=(self.size[0], self.size[1]),
                                               mode='nearest')
        mask_overlap = F.interpolate(mask_overlap, size=(self.size[0], self.size[1]), mode='nearest')
        mask_voteX = F.interpolate(mask_voteX, size=(self.size[0], self.size[1]), mode='nearest')
        mask_voteY = F.interpolate(mask_voteY, size=(self.size[0], self.size[1]), mode='nearest')
        mask_peaks = F.interpolate(mask_peaks, size=(self.size[0], self.size[1]), mode='nearest')
        mask_radius = F.interpolate(mask_radius, size=(self.size[0], self.size[1]), mode='nearest')

        image = image.squeeze(0)
        label = label.squeeze(0)
        mask_distanceTransform = mask_distanceTransform.squeeze(0)
        mask_overlap = mask_overlap.squeeze(0)
        mask_voteX = mask_voteX.squeeze(0)
        mask_voteY = mask_voteY.squeeze(0)
        mask_peaks = mask_peaks.squeeze(0)
        mask_radius = mask_radius.squeeze(0)
        image = self.TFNormalize(image)

        return image, label, mask_distanceTransform, mask_overlap, mask_voteX, mask_voteY, mask_peaks, mask_radius


class SquarePad:
    def __call__(self, image):
        max_wh = 400
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class PollenCls(Dataset):
    def __init__(self, dbinfo, size=[400, 400], set_name='train', maskType='pred'):
        if set_name == 'val':
            set_name = 'test'
        self.set_name = set_name
        self.imageList = dbinfo[set_name + '_cls_imgList']

        if maskType == 'pred':
            self.segList = dbinfo[set_name + '_cls_predMaskList']
        else:
            self.segList = dbinfo[set_name + '_cls_grndMaskList']

        self.clsIDList = dbinfo[set_name + '_cls_classID']
        self.transform = transform
        # self.size = size

        self.TFNormalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.current_set_len = len(self.imageList)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        # self.TFresize = transforms.Resize((self.size[0],self.size[1]))
        self.Pad = SquarePad()

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curImageName = self.imageList[idx]
        curMaskName = self.segList[idx]
        curLabelIndex = self.clsIDList[idx]  # .astype(np.float32)-1
        curLabelIndex = np.array([curLabelIndex]).astype(np.float32)

        image = PIL.Image.open(curImageName)
        image = self.Pad(image)
        segMask = PIL.Image.open(curMaskName)
        segMask = self.Pad(segMask)

        if self.set_name == 'train' and np.random.random(1) > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            segMask = segMask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        image = np.array(image)
        segMask = np.array(segMask).astype(np.float32)
        segMask = segMask // 255
        times = np.random.randint(4)
        image = np.rot90(image, times).copy()
        segMask = np.rot90(segMask, times).copy()

        # print('rot90 ', image.max(), image.min())

        image = self.TF2tensor(image)
        # print('to tensor ', image.max(), image.min())

        image = self.TFNormalize(image)
        # print('normalize ', image.max(), image.min())

        segMask = torch.from_numpy(segMask).unsqueeze(0)  # self.TF2tensor(mask_overlap)
        label = torch.from_numpy(curLabelIndex).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        image = image.unsqueeze(0)
        segMask = segMask.unsqueeze(0)

        # print(image.shape, segMask.shape)
        # image = F.interpolate(image, size=(self.size[0],self.size[1]), mode='bilinear', align_corners=True)
        # segMask = F.interpolate(segMask, size=(self.size[0], self.size[1]), mode='nearest')
        image = image.squeeze(0)
        label = label.squeeze(0)
        return image, segMask, label
