import json
from torch.utils.data import Dataset
import torch
import random
from .utils import load_datum, process_anyres_image, gen_grid_points, image_augment
import torch.distributed as dist

def check_nan(output):
    if torch.isnan(output).sum()>0:
        return 1
    else:
        return 0

class Classification_Dataset(Dataset):
    def __init__(self, dataset_name,data,crop_size,aug=False, patch_max=6,flag_2D=False):
        
        self.crop_size=crop_size
        self.data=data
        self.dataset_name=dataset_name

        self.aug = aug
        self.flag_2D = flag_2D
        self.grids = gen_grid_points(patch_max,crop_size,flag_2D)

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):

        sample = random.choices(self.data)[0]
        img_path=sample["img_path"][0] if len(sample["img_path"]) == 1 else sample["img_path"]
        cls_label=sample["disease_label"]

        image = load_datum(img_path, flag_2D=self.flag_2D)

        if self.aug:
            image = image_augment(image)

        image, _, voxel_mark, padded_size, patch_num = process_anyres_image(image, None, self.grids, self.crop_size,None,self.flag_2D)

        return {
            'image':image,
            'cls_label':cls_label,
            'mark':voxel_mark,
            'patch_num':patch_num
            }

class DataCollator_cls(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances):

        batch = {}
        batch['image'] = [instance['image'] for instance in instances]
        batch['mark'] = [instance['mark'] for instance in instances]
        batch['cls_label'] = [instance['cls_label'] for instance in instances]
        batch['patch_num'] = [instance['patch_num'] for instance in instances]

        return batch