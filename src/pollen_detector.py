from __future__ import print_function, division

import json
import logging
import os
import sys
import warnings  # ignore warnings
from datetime import datetime, timezone
from glob import glob

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from natsort import os_sorted
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage import measure
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.network_arch import PollenDet_SegDistTransform

logging.basicConfig(format='%(asctime)s %(levelname)-7s : %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("pollen_detector.py")

warnings.filterwarnings("ignore")
logger.info(sys.version)
logger.info(torch.__version__)


class PollenDetector:
    def __init__(self, model_file_path, crops_dir_path, detections_dir_path_prefix, num_processes, num_workers,
                 batch_size):
        self.model_file_path = model_file_path
        self.crops_dir_path = crops_dir_path

        # Get the current date and time
        now = datetime.now()
        # Format as a string
        datetime_postfix = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.detections_dir_path = detections_dir_path_prefix + "_" + datetime_postfix

        self.num_processes = num_processes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model = None
        self.dbinfo = None
        self.det_datasets = None
        self.data_loader = None
        self.conf_thresh = 0.20
        self.device = "cpu"
        self.tensor_size = [1024, 1024]  # set to crop size, to tell model what size tensor to expect
        logger.info("Available CPU Count: " + str(mp.cpu_count()))
        self.detections_list = []
        self.filtered_detections_list = []

    def initialize_model(self):
        logger.info("Initializing model")
        # cpu or cuda
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        logger.info(self.device)

        self.model = PollenDet_SegDistTransform(34, scaleList=[0], pretrained=False)
        self.model.encoder.encoder.conv1 = nn.Conv2d(27, 64, (7, 7), (2, 2), (3, 3),
                                                     bias=False)  # change dimensions of the first layer in the encoder
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        self.model.training = False

    def initialize_dataset(self):
        logger.info("Initializing dataset")
        self.det_datasets = PollenDet4Eval(path_to_image=self.crops_dir_path, dbinfo=self.dbinfo)

    def initialize_data_loader(self):
        logger.info("Initializing data loader")
        self.data_loader = DataLoader(self.det_datasets, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

    def generate_dbinfo(self):
        """
        Function to generate the alphabetically sorted dbinfo tuple list
        :return: None
        """
        logger.info("Generating dbinfo")
        dir_list = []
        with os.scandir(self.crops_dir_path) as crops_dir:
            main_folders = sorted((entry.name for entry in crops_dir if entry.is_dir()))
        main_folders = sorted(main_folders)
        for main_folder in main_folders:
            # TODO: This can be removed later.
            # with os.scandir(os.path.join(self.crops_dir_path, main_folder)) as main_fd:
            #     sub_folders = sorted((entry.name for entry in main_fd if entry.is_dir()))
            
            # Sort sub folders in natural order
            sub_folders = os_sorted(glob(os.path.join(self.crops_dir_path, main_folder, '*')))
            for sub_folder in sub_folders:
                dir_list.append((os.path.basename(main_folder), os.path.basename(sub_folder)))

        self.dbinfo = {"cropped_images_list": dir_list}

    @staticmethod
    def create_circular_mask(mask, center, radius, value=1):
        h, w = mask.shape[:2]
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

        tmp_mask = dist_from_center <= radius
        mask[tmp_mask] = value

        return mask

    @staticmethod
    def create_reverse_mask(mask, center, radius, value=1):
        h, w = mask.shape[:2]
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

        tmp_mask = dist_from_center > radius
        mask[tmp_mask] = value

        return mask

    @staticmethod
    def iou(box1, box2):
        """
        We assume that the box follows the format:
        box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
        where (x1,y1) and (x3,y3) represent the top left coordinate,
        and (x2,y2) and (x4,y4) represent the bottom right coordinate
        """
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        assert x1 < x2
        assert y1 < y2
        assert x3 < x4
        assert y3 < y4

        # determine the coordinates of the intersection rectangle
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

        # compute the area of both AABBs
        bb1_area = (x2 - x1) * (y2 - y1)
        bb2_area = (x4 - x3) * (y4 - y3)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.1
        return iou

    @staticmethod
    def nms(boxes, conf_threshold=0.1, iou_threshold=0.5):
        bbox_list_threshold_applied = []
        bbox_list_new = []
        bbox_list_new_txt = []

        # Stage 1: sort boxes, filter out boxes with low confidence
        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[1])
        for box in boxes_sorted:
            if box[1] > conf_threshold:
                bbox_list_threshold_applied.append(box)
        # Stage 2: loop through the boxes, remove boxes with high IoU
        while len(bbox_list_threshold_applied) > 0:
            current_box = bbox_list_threshold_applied.pop(0)
            bbox_list_new.append(current_box)
            current_box_txt = (
                current_box[0], str(current_box[1]), str(current_box[2]), str(current_box[3]), str(current_box[4]),
                str(current_box[5]), str(current_box[6]), str(current_box[7]))
            current_box_txt = ' '.join(current_box_txt)
            bbox_list_new_txt.append(current_box_txt)

            for box in bbox_list_threshold_applied:
                if current_box[0] == box[0]:
                    iou = PollenDetector.iou(current_box[8:12], box[8:12])
                    if iou > iou_threshold:
                        bbox_list_threshold_applied.remove(box)

        return bbox_list_new, bbox_list_new_txt

    def process_parallel(self):
        logger.info("Running in parallel mode")
        self.model.share_memory()

        # Set up the multiprocessing manager to store the list of detections
        manager = mp.Manager()
        self.detections_list = manager.list()

        worker_loaders = torch.utils.data.random_split(self.det_datasets, [
            1 / self.num_processes] * self.num_processes)
        processes = []
        for worker_id in range(self.num_processes):
            logger.info("Starting process: " + str(worker_id))
            p = mp.Process(target=PollenDetector.find_potential_pollen_detections,
                           args=(self, DataLoader(worker_loaders[worker_id], batch_size=self.batch_size,
                                                  worker_init_fn=worker_init_fn(worker_id),
                                                  num_workers=self.num_workers), worker_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def find_potential_pollen_detections(self, data_loader=None, process_id=None):
        logger.info("Processing crop images")

        if not os.path.exists(self.detections_dir_path):
            try:
                os.makedirs(self.detections_dir_path)
            except OSError:
                logger.error("Creation of the directory %s failed" % self.detections_dir_path)

        if data_loader is None:
            data_loader = self.data_loader

        iter_count, sample_count = 0, 0

        with torch.no_grad():
            # Iterate through batches
            for sample in data_loader:
                start_time = datetime.now()
                iter_count += 1

                if iter_count % 25 == 0:
                    logger.info('{}/{}'.format(iter_count, len(self.det_datasets)))

                cur_img, current_example = sample
                if process_id is not None:
                    logger.info(
                        "Process ID: " + str(process_id) + " Started processing crop images: " + str(current_example))
                else:
                    logger.info("Started processing crop images: " + str(current_example))
                cur_img = cur_img.to(self.device)

                outputs = self.model(cur_img)
                pred_seg = outputs[('segMask', 0)]
                pred_dist_transform_output = outputs[('output', 0)]
                pred_dist_transform_output = pred_dist_transform_output.squeeze().cpu().detach().numpy()
                softmax_output = pred_seg.squeeze().cpu().detach().numpy()

                # Iterate within a batch
                for index in range(cur_img.shape[0]):
                    if softmax_output.ndim == 3:
                        softmax = softmax_output[index]
                        pred_dist_transform = gaussian_filter(pred_dist_transform_output[index], sigma=10)
                    elif softmax_output.ndim == 2:
                        softmax = softmax_output
                        pred_dist_transform = gaussian_filter(pred_dist_transform_output, sigma=10)
                    else:
                        logger.error("Invalid softmax_output.ndim: " + str(softmax_output.ndim))
                        raise ValueError("Invalid softmax_output.ndim: " + str(softmax_output.ndim))

                    # find peaks, zero-out background noise
                    voting4center = np.copy(pred_dist_transform)
                    voting4center[voting4center < 0.001] = 0
                    coord_peaks = feature.peak_local_max(voting4center, min_distance=25,
                                                         exclude_border=False)  # originally min_distance =5, changed to 25

                    # create detection mask using peaks and predicted radius
                    pred_radius_list = []
                    size = (400, 400)

                    for i in range(coord_peaks.shape[0]):
                        # create full image detection mask
                        y, x = coord_peaks[i]

                        left = int(x - (size[0] / 2))
                        left = max(left, 0)
                        top = int(y - (size[0] / 2))
                        top = max(top, 0)
                        right = int(x + (size[0] / 2))
                        right = max(right, 0)
                        bottom = int(y + (size[0] / 2))
                        bottom = max(bottom, 0)

                        tmp_crop = softmax[top:bottom, left:right]
                        thresh = threshold_otsu(tmp_crop)
                        tmp_crop = tmp_crop > thresh  # binarize
                        tmp_crop = measure.label(tmp_crop, background=0)
                        props = measure.regionprops(tmp_crop)  # get the properties of the connected components

                        diameter = [prop.major_axis_length for prop in props]  # diameter for connected components
                        if len(diameter) != 0 and max(diameter) != 0:
                            radius = int(max(diameter) / 2)

                            pred_radius_list += [radius]
                            a = len(pred_radius_list) - 1

                            tmp_mask = voting4center * 0
                            tmp_mask = PollenDetector.create_circular_mask(tmp_mask, [y, x], pred_radius_list[a],
                                                                           value=1)

                            class_name = "det"
                            left_bb = x - radius
                            left_bb = max(left_bb, 0)
                            top_bb = y - radius
                            top_bb = max(top_bb, 0)
                            right_bb = x + radius
                            right_bb = max(right_bb, 0)
                            bottom_bb = y + radius
                            bottom_bb = max(bottom_bb, 0)

                            # Calculate global coordinates
                            coord_split_list = current_example[1][index].split('_')
                            tile_image_coordinate_x = int(coord_split_list[0].split('x')[0])
                            tile_image_coordinate_y = int(coord_split_list[1].split('y')[0])
                            left_bb_global = left_bb + tile_image_coordinate_x
                            top_bb_global = top_bb + tile_image_coordinate_y
                            right_bb_global = right_bb + tile_image_coordinate_x
                            bottom_bb_global = bottom_bb + tile_image_coordinate_y

                            masked_softmax = np.ma.masked_where(tmp_mask == 0, softmax)
                            confidence = np.nanmean(masked_softmax)
                            bbox_info2 = [class_name, confidence, left_bb, top_bb, right_bb, bottom_bb,
                                          current_example[0][index], current_example[1][index], left_bb_global,
                                          top_bb_global, right_bb_global, bottom_bb_global]

                            self.detections_list.append(bbox_info2)

                if process_id is not None:
                    logger.info(
                        "Process ID: " + str(process_id) + " Completed finding potential pollen detections on : " + str(
                            current_example))
                else:
                    logger.info("Completed finding potential pollen detections on : " + str(current_example))
                duration = datetime.now() - start_time
                logger.info("Duration: " + str(duration))

    def process_pollen_detections(self):
        # Apply non-max suppression
        self.filtered_detections_list = PollenDetector.nms(self.detections_list, conf_threshold=self.conf_thresh,
                                                           iou_threshold=0.3)
        self.filtered_detections_list = self.filtered_detections_list[0]

        # create detection mask and center mask using the information on each detection in filtered_detections_list
        det_mask = np.zeros((self.tensor_size[0], self.tensor_size[1]), dtype=np.float32)

        for i in range(len(self.filtered_detections_list)):

            ndpi_filename = self.filtered_detections_list[i][6]
            tile_image_coordinates = self.filtered_detections_list[i][7]

            logger.info(
                "Pollen detected: " + ndpi_filename + " " + tile_image_coordinates)

            # list filenames for images in the stack
            slice_paths = []
            current_image_path = os.path.join(self.crops_dir_path, ndpi_filename, tile_image_coordinates)
            for file in sorted(os.listdir(str(current_image_path))):
                if file.endswith('.png'):
                    slice_path = os.path.join(str(current_image_path), file)
                    slice_paths.append(slice_path)

            confidence = float(self.filtered_detections_list[i][1])
            left_bb = int(self.filtered_detections_list[i][2])
            top_bb = int(self.filtered_detections_list[i][3])
            right_bb = int(self.filtered_detections_list[i][4])
            bottom_bb = int(self.filtered_detections_list[i][5])
            diameter = max(right_bb - left_bb, bottom_bb - top_bb)
            radius = diameter / 2
            x = left_bb + radius
            y = top_bb + radius
            det_mask = PollenDetector.create_circular_mask(det_mask, [y, x], radius, value=i + 1)
            # crop detection mask
            crop_mask = det_mask[top_bb:bottom_bb, left_bb:right_bb]

            # crop image and stack slices together
            slices = []

            for idx in range(len(slice_paths)):
                img_slice = Image.open(slice_paths[idx])
                img_slice = np.array(img_slice)[top_bb:bottom_bb, left_bb:right_bb]
                slices.append(img_slice)

            img_path = self.detections_dir_path
            metadata: dict = dict()
            metadata["sample_filename"] = ndpi_filename
            metadata["tile_image_coordinates"] = tile_image_coordinates
            metadata["pollen_image_coordinates"] = "((" + str(left_bb) + "," + str(top_bb) + "), (" + str(
                right_bb) + "," + str(bottom_bb) + "))"
            metadata["confidence"] = confidence
            metadata["processed_datetime_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'

            k = 1
            img_path_2 = os.path.join(str(img_path), ndpi_filename + '_' + tile_image_coordinates + '_' + str(k))
            while os.path.exists(img_path_2):
                img_path_2 = os.path.join(str(img_path), ndpi_filename + '_' + tile_image_coordinates + '_' + str(k))
                k += 1

            for m in range(len(slices)):
                if not os.path.exists(img_path_2):
                    os.makedirs(img_path_2)
                img_filename = "{}/{}".format(img_path_2, str(m) + 'z.png')

                if isinstance(slices[m], np.ndarray):
                    img_slice = Image.fromarray(slices[m])
                    img_slice.save(img_filename)

            # save cropped mask
            mask_path = img_path_2

            k = 1
            mask_filename = "{}/{}_{}{}".format(mask_path, "mask", k, '.png')
            while os.path.exists(mask_filename):
                mask_filename = "{}/{}_{}{}".format(mask_path, "mask", k, '.png')
                k += 1

            if isinstance(crop_mask, np.ndarray):
                crop_mask = Image.fromarray((crop_mask * 255).astype(np.uint8))
            crop_mask.save(mask_filename)

            # save metadata
            metadata_path = img_path_2

            k = 1
            metadata_filename = "{}/{}_{}{}".format(metadata_path, "metadata", k, '.json')
            while os.path.exists(metadata_filename):
                metadata_filename = "{}/{}_{}{}".format(metadata_path, "metadata", k, '.json')
                k += 1

            with open(metadata_filename, "w") as metadata_json_file:
                json.dump(metadata, metadata_json_file, indent=2)


def worker_init_fn(worker_id):
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)


class PollenDet4Eval(Dataset):
    def __init__(self, path_to_image, dbinfo):

        self.path_to_image = path_to_image
        self.dbinfo = dbinfo
        self.sampleList = self.dbinfo['cropped_images_list']
        self.current_set_len = len(self.sampleList)
        self.TF2tensor = transforms.ToTensor()

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        current_example = self.sampleList[idx]
        current_image_path = os.path.join(str(self.path_to_image), current_example[0], current_example[1])

        imagestack_array = []
        for file in sorted(os.listdir(current_image_path)):
            if file.endswith('.png'):
                try:
                    slice_path = Image.open(os.path.join(current_image_path, file))
                    imagestack_array.append(np.asarray(slice_path).astype(np.float32) / 255)
                except IOError:
                    logger.error("Error reading image file: " + str(file) + " in " + str(current_image_path))
        image = np.block(imagestack_array)
        if image.shape[2] < 27:
            pad_val = 27 - image.shape[2]
            npad = ((0, 0), (0, 0), (0, pad_val))
            image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)

        image = self.TF2tensor(image)
        return image, current_example
