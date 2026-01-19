from torch.utils.data import Dataset
import torch
import numpy as np
import random
import os
from .utils import *

class Segmentation_trainDataset(Dataset):
    def __init__(self, dataset_name, data, crop_size=(256,256,32),aug=True, patch_max=8):
        
        self.crop_size=crop_size
        self.foreground_crop_prob=1.0
        self.label_based_crop_prob=0.5
        self.uncenter_prob=0.0
        self.pos_label_first_prob=1.0
        self.neg_label_ratio_threshold=0.5
        self.max_queries=16
        self.data=data
        self.dataset_name=dataset_name
        self.max_class=get_maxclass(data)
        self.top_label = 60
        self.random_pick_label = True if self.max_class > self.top_label else False
        self.max_class = self.top_label if self.random_pick_label else self.max_class
        self.aug = aug
        self.grids = gen_grid_points(patch_max,crop_size)
    
    def __len__(self):
        return 10000000
    
    def _select_pos_labels(self, label_index_ls, is_pos_ls, neg_label_ratio_threshold):
        """
        Args:
            label_index_ls (List of int) : candidate labels (channel index in segmentation mask)
            is_pos_ls (List of bool) : positive label (True) or not (False), equal length to label_index_ls
        
        Returns:
            chosen_label_index_ls (List of int) : chosen subset of label_index_ls
            chosen_is_pos (List of bool) : chosen subset of is_pos_ls
        """
        # divide all the labels into pos and neg
        pos_label_index_ls = []
        neg_label_index_ls = []
        for i, is_pos in zip(label_index_ls, is_pos_ls):
            if is_pos:
                pos_label_index_ls.append(i)
            else:
                neg_label_index_ls.append(i)
        pos_num = len(pos_label_index_ls)
        neg_num = len(neg_label_index_ls)
        
        if pos_num == 0:
            # degrad to random sample
            sample_num = min(self.max_queries, len(label_index_ls))
            chosen_label_index_ls = random.sample(label_index_ls, sample_num)
            chosen_is_pos = [False] * sample_num
            return chosen_label_index_ls, chosen_is_pos
        
        # indicate each sample is pos or neg
        chosen_is_pos = []
        
        if pos_num <= self.max_queries:
            # all pos labels are included, then sample some neg labels
            chosen_label_index_ls = pos_label_index_ls 
            chosen_is_pos += [True] * pos_num
            max_neg_num = int(neg_label_ratio_threshold * pos_num)    # neg label num < (pos label num) * x%
            left_pos_num = min(self.max_queries-pos_num, max_neg_num)   # neg label num < self.max_queries-pos_num
            if neg_num <= left_pos_num:
                # neg are all sampled
                chosen_label_index_ls += neg_label_index_ls
                chosen_is_pos += [False] * neg_num
            else:
                # neg are sampled to control the ratio and max label num
                chosen_label_index_ls += random.sample(neg_label_index_ls, left_pos_num)
                chosen_is_pos += [False] * left_pos_num
        else:
            # no neg labels are sampled
            chosen_label_index_ls = random.sample(pos_label_index_ls, self.max_queries)
            chosen_is_pos += [True] * self.max_queries

        return chosen_label_index_ls, chosen_is_pos

    
    def _crop(self, image, sample, roi_crop_prob, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            try:
                image, y1x1z1_y2x2z2 = self._roi_crop(image, sample, label_based_crop_prob)
            except:
                image, y1x1z1_y2x2z2 = self._random_crop(image)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, datum, label_based_crop_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        
        if random.random() < label_based_crop_prob:
        # if random.random() < 0:
            # find a pos label and crop based on it (ensure at least one pos label before roi crop
            pos_label_idx_ls = [i for i, t_or_f in enumerate(datum['renorm_y1x1z1_y2x2z2']) if t_or_f]
            pos_label_idx = random.sample(pos_label_idx_ls, 1)[0]
            mask_to_select = self._load_mask(datum, [datum['label'][pos_label_idx]], [datum['renorm_y1x1z1_y2x2z2'][pos_label_idx]])    # 1 h w d
            mask_to_select = mask_to_select[0, :, :, :]  # h w d 
        else:
            # crop based on all labels
            _, h, w, d = datum['chwd']
            mask_to_select = torch.zeros((h, w, d), dtype=torch.bool)
            y1, x1, z1, y2, x2, z2 = datum['roi_y1x1z1_y2x2z2']
            npy_path = f"{datum['renorm_segmentation_dir']}.npy"
            if os.path.exists(npy_path):
                mask_to_select[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(npy_path))
            else:
                npz_path = npy_path[:-4]+'.npz'
                mask_to_select[y1:y2, x1:x2, z1:z2] =torch.tensor(np.load(npz_path)['arr_0'])
        # select a voxel
        voxels_foreground = torch.nonzero(mask_to_select, as_tuple=True)   # (tensor(...), tensor(...), tensor(...))
        selected_index = random.randint(0, voxels_foreground[0].shape[0]-1)
        selected_voxel = (voxels_foreground[0][selected_index].item(), voxels_foreground[1][selected_index].item(), voxels_foreground[2][selected_index].item())
        
        if selected_voxel[0] - croph // 2 > 0:
            start_y = selected_voxel[0] - croph // 2
            if start_y + croph < imgh:
                end_y = start_y + croph
            else:
                end_y = imgh
                start_y = imgh-croph
        else:
            start_y = 0
            end_y = croph
            
        if selected_voxel[1] - cropw // 2 > 0:
            start_x = selected_voxel[1] - cropw // 2
            if start_x + cropw < imgw:
                end_x = start_x + cropw
            else:
                end_x = imgw
                start_x = imgw-cropw
        else:
            start_x = 0
            end_x = cropw

        if selected_voxel[2] - cropd // 2 > 0:
            start_z = selected_voxel[2] - cropd // 2
            if start_z + cropd < imgd:
                end_z = start_z + cropd
            else:
                end_z = imgd
                start_z = imgd-cropd
        else:
            start_z = 0
            end_z = cropd


        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]

        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _random_crop(self, image):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        # 
        start_y = random.randint(0, imgh - croph)
        end_y = start_y + croph
        start_x = random.randint(0, imgw - cropw)
        end_x = start_x + cropw
        start_z = random.randint(0, imgd - cropd)
        end_z = start_z + cropd
        #
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]
        
        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]

    def _load_mask(self, datum, labels_to_load, y1x1z1_y2x2z2_to_load):
        """
        Args:
            datum (dict): sample info (a line from jsonl file

        Returns:
            mc_mask: (N, h, w, d)
            labels: list of N str
            is_pos: lits of True/False
        """
        _, h, w, d = datum['chwd']
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in labels_to_load] # /remote-home/share/SAM/processed_files/MSD_Liver/segmentation/27/liver tumor.npy
        y1x1z1_y2x2z2_ls = y1x1z1_y2x2z2_to_load
        
        mc_mask = []
        is_pos = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d), dtype=torch.bool)
            # not empty, load and embed non-empty cropped_volume
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                if os.path.exists(mask_path):
                    mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
                else:
                    npz_path = mask_path[:-4]+'.npz'
                    mask[y1:y2, x1:x2, z1:z2]=torch.tensor(np.load(npz_path)['arr_0'])
                is_pos.append(True)
            else:
                is_pos.append(False)
            mc_mask.append(mask)
            
        mc_mask = np.stack(mc_mask, axis=0)   # n h w d
        
        return mc_mask

    def is_overlap(self, a_y1x1z1_y2x2z2, b_y1x1z1_y2x2z2):
        # judge is overlap or not between two cubes
        a_y1, a_x1, a_z1, a_y2, a_x2, a_z2 = a_y1x1z1_y2x2z2
        b_y1, b_x1, b_z1, b_y2, b_x2, b_z2 = b_y1x1z1_y2x2z2
        overlap_x = not (a_x2 < b_x1 or b_x2 < a_x1)
        overlap_y = not (a_y2 < b_y1 or b_y2 < a_y1)
        overlap_z = not (a_z2 < b_z1 or b_z2 < a_z1)

        return overlap_x and overlap_y and overlap_z
    
    def _find_pos_labels_in_crop(self, crop_y1x1z1_y2x2z2, labels_y1x1z1_y2x2z2):
        is_pos = []
        for y1x1z1_y2x2z2 in labels_y1x1z1_y2x2z2:
            if y1x1z1_y2x2z2 and self.is_overlap(y1x1z1_y2x2z2, crop_y1x1z1_y2x2z2):
                is_pos.append(True)
            else:
                is_pos.append(False)
        return is_pos
    
    def __getitem__(self, idx):

        # random choose a sample
        sample = random.choices(self.data)[0]

        if "npz" in sample['renorm_image']:
            image=np.load(sample['renorm_image'])['arr_0']
        else:
            if os.path.exists(sample['renorm_image']):
                image=np.load(sample['renorm_image'])
            else:
                npz_path = sample['renorm_image'][:-4]+'.npz'
                image=np.load(npz_path)['arr_0']

        # so these are the chosen labels  


        if (not self.random_pick_label) or len(sample['label'])<=self.top_label:
            chosen_label = sample['label']
            chosen_y1x1z1_y2x2z2 = sample['renorm_y1x1z1_y2x2z2']
            chosen_label_idx = range(len(chosen_label))
        else:
            chosen_label_idx = random.sample(range(len(sample['label'])), self.top_label)
            chosen_label = [sample['label'][label_idx] for label_idx in chosen_label_idx]
            chosen_y1x1z1_y2x2z2 = [sample['renorm_y1x1z1_y2x2z2'][label_idx] for label_idx in chosen_label_idx]
        
        mask = self._load_mask(sample, chosen_label, chosen_y1x1z1_y2x2z2)

        if self.aug:
            image, mask = seg_augment(image, mask)

        image, mask, mark, padded_size, patch_num = process_anyres_image(image, mask, self.grids, self.crop_size) # self.augmentator

        # image, mark, mask, _ = resize_and_pad_image(image, mask, (128,128,128))

        mask = pad_mask(torch.tensor(mask),self.max_class)

        return {
            'image':image,
            'mask':mask.int().float(),
            'patch_num':patch_num,
            'chosen_label_idx':chosen_label_idx
        }

class DataCollator_seg(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances):

        batch = {}
        batch['image'] = [instance['image'] for instance in instances]
        batch['mask'] = [instance['mask'] for instance in instances]
        batch['patch_num'] = [instance['patch_num'] for instance in instances]
        batch['chosen_label_idx'] = [instance['chosen_label_idx'] for instance in instances]

        return batch
    