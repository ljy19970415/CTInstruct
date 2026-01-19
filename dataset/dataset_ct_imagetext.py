from torch.utils.data import Dataset
import numpy as np
import random
import numpy as np
from .utils import load_datum, load_from_bucket, gen_grid_points, image_augment
from .utils import process_anyres_image
from time import time

class imageText_Dataset(Dataset):
    def __init__(self, dataset_name,data,crop_size=(256,256,32),aug=False, patch_max = 8):
        
        self.crop_size=crop_size
        self.data=data
        self.dataset_name=dataset_name
        self.grids = gen_grid_points(patch_max,crop_size)
        self.aug = aug
    
    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        # random choose a sample
        sample = random.choices(self.data)[0]
        img_path=sample["image_path"][0]
        answer=sample["answer"]

        if 's3://zhangxiaoman_hdd_new/DATA/M3D' in img_path:
            image = load_from_bucket(img_path)
        else:
            image = load_datum(img_path,mask_path=None,flag_2D=False)

        if self.aug:
            image = image_augment(image)

        image, target, voxel_mark, padded_size, patch_num = process_anyres_image(image, None, self.grids, self.crop_size)

        return {
            'image':image,
            'text':answer,
            'mark':voxel_mark,
            'patch_num':patch_num,
            'idx':idx
        }

class DataCollator_imagetext(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances):

        batch = {}
        batch['image'] = [instance['image'] for instance in instances]
        batch['mark'] = [instance['mark'] for instance in instances]
        batch['text'] = [instance['text'] for instance in instances]
        batch['patch_num'] = [instance['patch_num'] for instance in instances]
        batch['idx'] = [instance['idx'] for instance in instances]

        return batch


class imageText_Retrieval_Dataset(Dataset):
    def __init__(self, dataset_name,data,crop_size=(256,256,32),aug=False,batch_size=1, patch_max = 8):
        
        self.crop_size=crop_size
        self.batch_size = batch_size
        self.data=data
        self.dataset_name=dataset_name
        self.grids = gen_grid_points(patch_max,crop_size)
        self.aug = aug
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # random choose a sample
        sample = self.data[idx]
        
        if "image_path" not in sample:
            img_path = sample['img_path'][0]
        else:
            img_path=sample["image_path"][0]
        answer=sample["answer"]
     
        if 's3://zhangxiaoman_hdd_new/DATA/M3D' in img_path:
            image = load_from_bucket(img_path)
        else:
            image = load_datum(img_path,mask_path=None,flag_2D=False)

        if self.aug:
            image = image_augment(image)

        image, target, voxel_mark, padded_size, patch_num = process_anyres_image(image, None, self.grids, self.crop_size)

        
        return {
            'image':image,
            'text':answer,
            'mark':voxel_mark,
            'patch_num':patch_num,
            'idx':idx
        }