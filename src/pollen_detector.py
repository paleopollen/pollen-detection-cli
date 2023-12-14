from __future__ import print_function, division

import sys
import warnings  # ignore warnings
import json

from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage import measure
from skimage.filters import threshold_otsu

from utils.dataset import *
from utils.network_arch import *
from utils.trainval_detSegDistTransform import *

warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)


class PollenDetector:
    def __init__(self, model_dir_path, crops_dir_path, detections_dir_path):
        self.model_dir_path = model_dir_path
        self.crops_dir_path = crops_dir_path
        self.detections_dir_path = detections_dir_path
        self.model = None
        self.dbinfo = None
        self.det_datasets = None
        self.data_loader = None
        self.conf_thresh = 0.013481323
        self.device = "cpu"
        self.tensor_size = [1024, 1024]  # set to crop size, to tell model what size tensor to expect

    def initialize_model(self):
        # cpu or cuda
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        print(self.device)

        self.model = PollenDet_SegDistTransform(34, scaleList=[0], pretrained=False)
        self.model.encoder.encoder.conv1 = nn.Conv2d(27, 64, (7, 7), (2, 2), (3, 3),
                                                     bias=False)  # change dimensions of the first layer in the encoder
        self.model.load_state_dict(torch.load(path.join(self.model_dir_path, 'bestValModel_encoder.paramOnly'),
                                              map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        self.model.training = False

    def initialize_data(self):
        path_to_mask = (
            '/Users/sandeep/Repositories/PALYIM/open_world_pollen_detection/03_NDPI_Slide_Annotation/Output/C3'
            '/C3_masks')
        set_name = 'test'  # 'test'
        self.det_datasets = PollenDet4Eval(path_to_image=self.crops_dir_path, path_to_mask=path_to_mask,
                                           dbinfo=self.dbinfo, size=self.tensor_size, set_name=set_name)

        self.data_loader = DataLoader(self.det_datasets, batch_size=1, shuffle=False, num_workers=0)

    def generate_dbinfo(self):
        """
        Function to generate the alphabetically sorted dbinfo tuple list
        :return: None
        """
        dir_list = []
        with os.scandir(self.crops_dir_path) as crops_dir:
            main_folders = sorted((entry.name for entry in crops_dir if entry.is_dir()))
        main_folders = sorted(main_folders)
        for main_folder in main_folders:
            with os.scandir(os.path.join(self.crops_dir_path, main_folder)) as main_fd:
                sub_folders = sorted((entry.name for entry in main_fd if entry.is_dir()))
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
        # detMask_info = detMask_info

        # Stage 1: sort boxes, filter out boxes with low confidence
        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[1])
        for box in boxes_sorted:
            if box[1] > conf_threshold:
                bbox_list_threshold_applied.append(box)
            else:
                pass
        # Stage 2: loop through the boxes, remove boxes with high IoU
        while len(bbox_list_threshold_applied) > 0:
            current_box = bbox_list_threshold_applied.pop(0)
            bbox_list_new.append(current_box)
            current_box_txt = (
                current_box[0], str(current_box[1]), str(current_box[2]), str(current_box[3]), str(current_box[4]),
                str(current_box[5]))
            current_box_txt = ' '.join(current_box_txt)
            bbox_list_new_txt.append(current_box_txt)

            for box in bbox_list_threshold_applied:
                if current_box[0] == box[0]:
                    iou = PollenDetector.iou(current_box[2:], box[2:])
                    # print(iou)
                    if iou > iou_threshold:
                        bbox_list_threshold_applied.remove(box)
                        # detMask_info.remove()

        return bbox_list_new, bbox_list_new_txt

    def process_crop_images(self):

        detections_dir = os.path.join(self.detections_dir_path)
        if not os.path.exists(detections_dir):
            os.makedirs(detections_dir)

        iter_count, sample_count = 0, 0
        for sample in self.data_loader:
            iter_count += 1

            print("Processing crop image: " + str(iter_count))

            if iter_count % 25 == 0:
                print('{}/{}'.format(iter_count, len(self.det_datasets)))

            cur_img, cur_mask_org_size, current_example = sample
            cur_img, cur_mask_org_size = cur_img.to(self.device), cur_mask_org_size.to(self.device)

            outputs = self.model(cur_img)
            pred_seg = outputs[('segMask', 0)]

            pred_dist_transform = outputs[('output', 0)]

            softmax = pred_seg

            # GT seg mask
            curmask_org_size = cur_mask_org_size.squeeze().cpu().detach().numpy()
            curmask_org_size_binary = curmask_org_size > 0

            # prediction:
            # create a list of (800x800) prediction distance transforms crops and softmax crops

            pred_dist_transform_crops = []
            softmax_crops = []

            for idx in range(0, 1):
                tmp_img = pred_dist_transform[idx, :, :, :].squeeze().cpu().detach().numpy()
                pred_dist_transform_crops.append(tmp_img)

            for idx in range(0, 1):
                tmp_img = softmax[idx, :, :, :].squeeze().cpu().detach().numpy()
                softmax_crops.append(tmp_img)

            # create full-sized pred distance transform
            mask_org_size = cur_mask_org_size.squeeze().cpu().detach().numpy()

            tmp_pred_dist_transform_1 = np.zeros_like(mask_org_size).astype(np.float32)

            tmp_pred_dist_transform_1[0:self.tensor_size[0], 0:self.tensor_size[0]] = pred_dist_transform_crops[0]

            pred_dist_transform = np.maximum.reduce(
                [tmp_pred_dist_transform_1])
            pred_dist_transform = gaussian_filter(pred_dist_transform, sigma=10)  # gaussian blur to get rid of shadow
            pred_distance_transform = np.copy(pred_dist_transform)

            tmp_softmax_1 = np.zeros_like(mask_org_size).astype(np.float32)

            tmp_softmax_1[0:self.tensor_size[0], 0:self.tensor_size[0]] = softmax_crops[0]

            softmax = np.nanmean(np.array([tmp_softmax_1]), axis=0)

            # find peaks, zero-out background noise
            voting4center = np.copy(pred_distance_transform)
            voting4center[voting4center < 0.001] = 0
            coord_peaks = feature.peak_local_max(voting4center, min_distance=25,
                                                 exclude_border=False)  # originally min_distance =5, changed to 25

            # list filenames for images in the stack
            slice_paths = []
            current_image_path = os.path.join(self.crops_dir_path, current_example[0][0], current_example[1][0])
            for file in sorted(os.listdir(str(current_image_path))):
                if file.endswith('.png'):
                    slice_path = os.path.join(str(current_image_path), file)
                    slice_paths.append(slice_path)

            # create detection mask using peaks and predicted radius
            pred_radius_list = []
            detection_info2 = []

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
                    tmp_mask = PollenDetector.create_circular_mask(tmp_mask, [y, x], pred_radius_list[a], value=1)

                    class_name = "det"
                    left_bb = x - radius
                    left_bb = max(left_bb, 0)
                    top_bb = y - radius
                    top_bb = max(top_bb, 0)
                    right_bb = x + radius
                    right_bb = max(right_bb, 0)
                    bottom_bb = y + radius
                    bottom_bb = max(bottom_bb, 0)

                    masked_softmax = np.ma.masked_where(tmp_mask == 0, softmax)
                    confidence = np.nanmean(masked_softmax)
                    bbox_info2 = [class_name, confidence, left_bb, top_bb, right_bb, bottom_bb]

                    detection_info2.append(bbox_info2)

            # Apply non-max suppression
            nms_bb = PollenDetector.nms(detection_info2, conf_threshold=self.conf_thresh, iou_threshold=0.3)
            nms_bb = nms_bb[0]

            # create detection mask and center mask using the information on each detection in [NMS_bb]
            det_mask = voting4center * 0

            # TODO: Pollen on 0x_16384y - Store confidence and detection coordinates within the 1024 by 1024 crop as
            #  text/JSON

            for i in range(len(nms_bb)):
                confidence = float(nms_bb[i][1])
                left_bb = int(nms_bb[i][2])
                top_bb = int(nms_bb[i][3])
                right_bb = int(nms_bb[i][4])
                bottom_bb = int(nms_bb[i][5])
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
                    img_slice = PIL.Image.open(slice_paths[idx])
                    img_slice = np.array(img_slice)[top_bb:bottom_bb, left_bb:right_bb]
                    slices.append(img_slice)

                # TODO: Please make sure to save pollen sub-crops and masks, and a text file of sample name,
                #  crop coordinates, pollen coordinates, and confidence scores! save slices of cropped image
                img_path = detections_dir

                metadata: dict = dict()
                metadata["sample_filename"] = current_example[0][0]
                metadata["crop_image_coordinates"] = current_example[1][0]
                metadata["pollen_image_coordinates"] = "((" + str(left_bb) + "," + str(top_bb) + "), (" + str(
                    right_bb) + "," + str(bottom_bb) + "))"
                metadata["confidence"] = str(confidence)

                k = 1
                img_path_2 = os.path.join(img_path, current_example[0][0] + '_' + current_example[1][0] + '_' + str(k))
                while os.path.exists(img_path_2):
                    img_path_2 = os.path.join(img_path,
                                              current_example[0][0] + '_' + current_example[1][0] + '_' + str(k))
                    k += 1

                for m in range(len(slices)):
                    if not os.path.exists(img_path_2):
                        os.makedirs(img_path_2)
                    img_filename = "{}/{}".format(img_path_2, str(m) + 'z.png')

                    if isinstance(slices[m], np.ndarray):
                        img_slice = PIL.Image.fromarray(slices[m])
                        img_slice.save(img_filename)

                # save cropped mask
                mask_path = img_path_2

                k = 1
                mask_filename = "{}/{}_{}{}".format(mask_path, "mask", k, '.png')
                while os.path.exists(mask_filename):
                    mask_filename = "{}/{}_{}{}".format(mask_path, "mask", k, '.png')
                    k += 1

                if isinstance(crop_mask, np.ndarray):
                    crop_mask = PIL.Image.fromarray((crop_mask * 255).astype(np.uint8))
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


class PollenDet4Eval(Dataset):
    def __init__(self, path_to_image, path_to_mask, dbinfo, size, set_name='test'):

        self.path_to_image = path_to_image
        self.path_to_mask = path_to_mask
        self.transform = transform
        self.dbinfo = dbinfo
        if set_name == 'val':
            set_name = 'test'
        self.set_name = set_name
        self.size = size
        self.resizeFactor = size[0] / 1000

        self.sampleList = self.dbinfo['cropped_images_list']

        self.TFNormalize = transforms.Normalize([0.5] * 27, [0.5] * 27)
        self.current_set_len = len(self.sampleList)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFresize = transforms.Resize((self.size[0], self.size[1]))

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        current_example = self.sampleList[idx]
        current_image_path = os.path.join(self.path_to_image, current_example[0], current_example[1])
        current_dist_transform_path = os.path.join(self.path_to_mask, "PAL1999_C3_sample22_slide1_5", "0x_1024y")
        mask = None

        imagestack_array = []
        for file in sorted(os.listdir(current_image_path)):
            if file.endswith('.png'):
                slice_path = Image.open(os.path.join(current_image_path, file))
                imagestack_array.append(np.asarray(slice_path).astype(np.float32) / 255)
        image = np.block(imagestack_array)
        if image.shape[2] < 27:
            pad_val = 27 - image.shape[2]
            npad = ((0, 0), (0, 0), (0, pad_val))
            image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)

        for file in os.listdir(current_dist_transform_path):
            if file.endswith('.png'):
                mask = Image.open(os.path.join(current_dist_transform_path, file))
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=2)

        label_org_size = np.copy(mask)
        label_org_size = torch.from_numpy(label_org_size).unsqueeze(0).unsqueeze(0).squeeze(4)

        image = self.TF2tensor(image)

        label_org_size = label_org_size.squeeze(0)

        return image, label_org_size, current_example


if __name__ == '__main__':
    pollen_detector = PollenDetector("model",
                                     "/Users/sandeep/Repositories/PALYIM/open_world_pollen_detection"
                                     "/03_NDPI_Slide_Annotation/Output/C3/C3_tiles", "detections")
    pollen_detector.generate_dbinfo()
    pollen_detector.initialize_data()
    pollen_detector.initialize_model()
    pollen_detector.process_crop_images()
