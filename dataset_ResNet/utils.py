import torch
import numpy as np
import monai
from einops import rearrange
from collections import OrderedDict
import torch.nn.functional as F
import math
import random
from io import BytesIO
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates

import monai.transforms as mt
# from petrel_client.client import Client
# client = Client('~/petreloss.conf') # client搭建了和ceph通信的通道
import os
from time import time

def rgb2gray(rgb):

    flag = True
    if len(rgb.shape) == 3:
        z_len = rgb.shape[-1]
        for i in range(z_len-1):
            if not torch.equal(rgb[:,:,i],rgb[:,:,i+1]):
                flag = False
                break
        
        if flag:
            return rgb[:,:,0:1]
            
        if rgb.shape[-1] == 3:
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif rgb.shape[-1] == 4:
            r, g, b, a = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], rgb[:,:,3]
            white = torch.ones_like(r)
            r = r * a + white * (1 - a)
            g = g * a + white * (1 - a)
            b = b * a + white * (1 - a)
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        gray = gray.unsqueeze(-1)
    
    else:
        gray = rgb.unsqueeze(-1)

    return gray


monai_loader_2d_mask = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image', 'label']),
                    monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                    # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                    monai.transforms.ToTensord(keys=["image", "label"]),
                ]
            )
monai_loader_3d_mask = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image', 'label']),
                    monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                    monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                    monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
                    monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                    monai.transforms.ToTensord(keys=["image", "label"]),
                ]
            )


monai_loader_2d = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image']),
                    monai.transforms.Lambdad(
                        keys='image',
                        func=lambda x: rgb2gray(x)
                    ),
                    monai.transforms.EnsureChannelFirstd(keys=["image"],channel_dim="no_channel"),
                    monai.transforms.ToTensord(keys=["image"]),
                ]
            )

monai_loader_3d = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image']),
                    monai.transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
                    monai.transforms.ToTensord(keys=["image"]),
                ]
            )

from monai.transforms import MapTransform

class ExtractCTd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            npz_dict = np.load(d[key])
            d[key] = npz_dict['ct']
        return d

monai_loader_3d_npz = monai.transforms.Compose(
                [
                    ExtractCTd(keys=['image']),
                    monai.transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
                    monai.transforms.ToTensord(keys=["image"]),
                ]
            )

def gen_grid_points(max_cnt, crop_size, flag_2D=False):
    # 256, 256, 32*(1-30)
    # 512 512 32*(1-12)
    # 512 256 32*(1-24)
    # 256 512 32*(1-24)
    # 768 256 32*(1-16)
    # 256 768 32*(1-16)
    # 512 768 32*(1-8)
    # 768 512 32*(1-8)
    # max_scale = max_cnt // batch_size if not ignore_batch_size else max_cnt
    max_scale = max_cnt
    grids = []
    if flag_2D:
        for i in range(1,10):
            for j in range(1, (max_scale//i)+1):
                grids.append([crop_size[0]*i,crop_size[1]*j,1])
    else:
        for i in range(1,10):
            for j in range(1,10):
                for z in range(1, (max_scale//(i*j))+1):
                    grids.append([crop_size[0]*i,crop_size[1]*j,crop_size[2]*z])
    return grids

def pad_mask(mask,max_class):
    
    if max_class != 1:

        class_num = mask.shape[0]

        pad = (0, 0, 0, 0, 0, 0, 0, max_class-class_num)
        mask = F.pad(mask, pad, 'constant', 0)   # nhwd

    return mask

def Normalization(torch_image, image_type):
    np_image = torch_image.numpy()
    if np.max(np_image) - np.min(np_image) > 1000:
        lower_bound, upper_bound = np.percentile(np_image, 0.5), np.percentile(np_image, 99.5)
        np_image = np.clip(np_image, lower_bound, upper_bound)
    np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    return np_image

def get_maxclass(samples):
    max_class=1
    for sample in samples:
        if len(sample["label"])>max_class:
            max_class=len(sample["label"])
    return max_class

def load_mask(mask_path, shape, choose_label, renorm_box):
    mask = torch.zeros(shape)
    mask_paths = [f"{mask_path}/{label}.npy" if os.path.exists(f"{mask_path}/{label}.npy") else f"{mask_path}/{label}.npz" for label in choose_label] # /remote-home/share/SAM/processed_files/Challenge_4C2021/segmentation/27/laryngeal cancer or hypopharyngeal cancer.npy
    for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, renorm_box):
        y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
        if "npz" in mask_path:
            mask[:,y1:y2, x1:x2, z1:z2]=torch.tensor(np.load(mask_path)['arr_0'])
        elif 'npy' in mask_path:
            mask[:,y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
    return mask

def load_datum(img_path,mask_path=None,flag_2D=False):
    if mask_path is not None:  # classification or segmentation
        if flag_2D:   # 2d image or 3d volumn
            dictionary = monai_loader_2d_mask({'image':img_path, 'label':mask_path})    
        else:
            dictionary = monai_loader_3d_mask({'image':img_path, 'label':mask_path})
        
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]

        mask = torch.where(mask>0.5, 1.0, 0.0)
        img = Normalization(img, 'CT')

        return img, mask
    else:
        if flag_2D:
            dictionary = monai_loader_2d({'image':img_path})
        elif isinstance(img_path, str) and '.npz' in img_path.split('/')[-1]:
            dictionary = monai_loader_3d_npz({'image':img_path})
        else:
            dictionary = monai_loader_3d({'image':img_path})

        img = dictionary['image'] # [1, H, W, D]
        
        if 'Radio_VQA' in img_path or 'RAD-chestCT' in img_path:
            img = img.permute(0,2,3,1)

        img = Normalization(img, 'CT')

        return img

train_transform = monai.transforms.Compose(
    [
        monai.transforms.RandRotate90(prob=0.5, spatial_axes=(0,1)),
        monai.transforms.RandFlip(prob=0.1, spatial_axis=0),
        monai.transforms.RandFlip(prob=0.1, spatial_axis=1),
        monai.transforms.RandFlip(prob=0.1, spatial_axis=2),
        monai.transforms.RandScaleIntensity(factors=0.1, prob=0.5),
        monai.transforms.RandShiftIntensity(offsets=0.1, prob=0.5),
    ]
)

seg_mask_transform = monai.transforms.Compose([
    monai.transforms.RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=(0, 1)),
    monai.transforms.RandGaussianNoised(keys='image', prob=0.2, mean=0.0, std=0.05),
    monai.transforms.RandScaleIntensityd(keys='image', factors=0.1, prob=0.5),
    monai.transforms.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.5),
    monai.transforms.RandAdjustContrastd(keys='image', prob=0.2, gamma=(0.7, 1.5))
])

def image_augment(image):

    return train_transform(image)

def seg_augment(image, mask):

    res = seg_mask_transform({'image':image, 'mask':mask})
    return res['image'], res['mask']

def load_from_bucket(url):
    monai_loader = monai.transforms.Compose(
        [
            monai.transforms.Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=("bilinear")),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    data = client.get(url)
    value_buf = memoryview(data)
    iostr = BytesIO(value_buf)
    img_array = np.load(iostr)

    img_array = np.transpose(img_array,(0,2,3,1))

    dictionary = monai_loader({'image':img_array})
    img = dictionary['image'] # [1, H, W, D]
    img = Normalization(img, 'CT')

    return img

def load_from_bucket_m3d(url):

    data = client.get(url)
    value_buf = memoryview(data)
    iostr = BytesIO(value_buf)
    img_array = np.load(iostr)

    return img_array

m3d_transform = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image']),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        
    ])
m3d_resize = monai.transforms.Compose([
    monai.transforms.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])

def load_local_m3d(image_path):

    if 'RAD-chestCT' in image_path:
        image = np.load(image_path)['ct'][None]
    elif 'CT-RATE' in image_path:
        dictionary = m3d_transform({"image":image_path})
        image = dictionary['image']
        image = np.transpose(image,(2,0,1))[None]
        
    
    image = m3d_resize(image)
    image = image - image.min()
    image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

    return image

def generate_random_crop_size(H, W, D, crop_ratio=0.04, max_attempts=1000):
    V = H * W * D
    target_volume = int(round(crop_ratio * V))
    
    if target_volume <= 0:
        raise ValueError("The target volume size is too small.")
    if target_volume > V:
        raise ValueError("The target volume size is too big.")
    
    for attempt in range(max_attempts):
        H1 = random.randint(1, min(H, target_volume))
        remaining_volume_after_H = target_volume // H1
        if remaining_volume_after_H == 0:
            continue

        W1 = random.randint(1, min(W, remaining_volume_after_H))
        remaining_volume_after_W = target_volume // (H1 * W1)
        if remaining_volume_after_W == 0:
            continue
        
        D1 = target_volume // (H1 * W1)

        if H1 * W1 * D1 == target_volume and D1 <= D:
            return H1, W1, D1
    
    scale = crop_ratio ** (1/3)
    H1 = max(1, int(round(H * scale)))
    W1 = max(1, int(round(W * scale)))
    D1 = max(1, int(round(D * scale)))

    return  H1, W1, D1

def check_2D(img_path):
    if '.jpg' in img_path or '.jpeg' in img_path or '.png' in img_path or '.mat' in img_path:
        return True
    else:
        return False

def nnUNet_resize(data, new_shape, do_separate_z=True, is_seg=False, axis=2, order=3, order_z=0):
    assert len(data.shape) == 3, "data must be (x, y, z)"
    assert len(new_shape) == len(data.shape)

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
        order = 1
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_data = []
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                else:
                    reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
            reshaped_data = np.stack(reshaped_data, axis)
            
            if shape[axis] != new_shape[axis]:

                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                reshaped_data = map_coordinates(reshaped_data, coord_map, order=order_z, mode='nearest').astype(dtype_data)
        else:
            reshaped_data = resize_fn(data, new_shape, order, **kwargs).astype(dtype_data)
        return reshaped_data.astype(dtype_data)
    else:
        return data

def resize_and_pad_image(image, target, target_resolution, flag_2D=False):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image: The input 2D/3D image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        The resized and padded image.
    """

    original_width, original_height, original_depth = image[0].shape[0], image[0].shape[1], image[0].shape[2] 
    target_width, target_height, target_depth = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_d = target_depth / original_depth

    if flag_2D:
        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)
        new_depth = original_depth
    else:

        if scale_w < scale_h and scale_w < scale_d:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
            new_depth = min(math.ceil(original_depth * scale_w), target_depth)
        elif scale_h < scale_w and scale_h < scale_d:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)
            new_depth = min(math.ceil(original_depth * scale_w), target_depth)
        else:
            new_depth = target_depth
            new_height = min(math.ceil(original_height * scale_d), target_height)
            new_width = min(math.ceil(original_width * scale_d), target_width)
    
    # Resize the image
    # resized_image = nnUNet_resize(image[0], (new_width, new_height, new_depth))
    image_resampler = mt.Resize(
        spatial_size=(new_width, new_height, new_depth), 
        mode="trilinear",  # 或 "bilinear"/"area"/"bicubic"
        align_corners=True  # 保持对齐（推荐启用）
    )

    resized_image = image_resampler(image)
    
    anyres_size = list(resized_image.shape)


    if target is not None:
        # MONAI handles batch inputs and GPU acceleration
        resampler = mt.Resize(spatial_size=(new_width, new_height, new_depth), mode='nearest')
        resized_target = resampler(target)  # Input shape: (17, 248, 522, 64)

    voxel_mark = np.ones(resized_image.shape[-1])[None,:]

    pad_x_left = (target_width - new_width) // 2
    pad_x_right = target_width - new_width - pad_x_left

    pad_y_left = (target_height - new_height) // 2
    pad_y_right = target_height - new_height - pad_y_left

    pad_z_left = (target_depth - new_depth) // 2
    pad_z_right = target_depth - new_depth - pad_z_left

    new_image = np.pad(resized_image, (
        (0,0),
        (pad_x_left, pad_x_right),
        (pad_y_left, pad_y_right),
        (pad_z_left, pad_z_right)),
        'constant', **{'constant_values': 0})

    voxel_mark = np.pad(voxel_mark, (
        (0,0),
        (pad_z_left, pad_z_right)),
        'constant', **{'constant_values': 0})
    
    if target is not None:
        new_target = np.pad(resized_target, (
        (0,0),
        (pad_x_left, pad_x_right),
        (pad_y_left, pad_y_right),
        (pad_z_left, pad_z_right)),
        'constant', **{'constant_values': -1})
    else:
        new_target = None

    return new_image, voxel_mark, new_target, anyres_size

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    original_width, original_height, original_depth = original_size[1], original_size[2], original_size[3]
    for width, height, depth in possible_resolutions:
        scale = min(width / original_width, height / original_height, depth / original_depth)
        downscaled_width, downscaled_height, downscaled_depth = int(original_width * scale), int(original_height * scale), int(original_depth * scale)
        effective_resolution = min(downscaled_width * downscaled_height * downscaled_depth, original_width * original_height* original_depth)
        wasted_resolution = (width * height * depth) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height, depth)

    return best_fit

def divide_to_patches(image, target, voxel_mark, patch_size, flag_2D):
    """
    Divides an image into patches of a specified size.

    Args:
        image: The input image (1, h, w, d).
        voxel_mark: attention mask of image (1, h, w, d) same as image shape
        target: The input mask (c, h w d)
        patch_size (int): input height of vision encoder

    Returns:
        list: A list of patches list. d N_patch patch_size patch_size
    """
    
    # assert image.shape == voxel_mark.shape, "image and voxel_mark shape don't match!"

    cls_num = len(image)

    if flag_2D:

        w_p = image.shape[1]//patch_size[0]
        h_p = image.shape[2]//patch_size[1]
        z_p = 1

        # image = image.view(cls_num, w_p, patch_size[0], h_p, patch_size[1], z_p, patch_size[2])
        image = rearrange(image, 'c (w x) (h y) (d z) -> c w x h y d z',x=patch_size[0],y=patch_size[1],z=1)   # B*C, H*W*D
        # image = image.permute(0, 5, 1, 3, 2, 4, 6)
        image = rearrange(image, 'c w x h y d z -> c d w h x y z')
        image = image.reshape(cls_num, z_p*w_p*h_p, patch_size[0], patch_size[1], 1)

        voxel_mark = rearrange(voxel_mark, 'c (d z) -> c d z',z=1)
        voxel_mark = np.repeat(voxel_mark, w_p*h_p, axis=1)

    else:

        w_p = image.shape[1]//patch_size[0]
        h_p = image.shape[2]//patch_size[1]
        z_p = image.shape[3]//patch_size[2]

        # image = image.view(cls_num, w_p, patch_size[0], h_p, patch_size[1], z_p, patch_size[2])
        image = rearrange(image, 'c (w x) (h y) (d z) -> c w x h y d z',x=patch_size[0],y=patch_size[1],z=patch_size[2])   # B*C, H*W*D
        # image = image.permute(0, 5, 1, 3, 2, 4, 6)
        image = rearrange(image, 'c w x h y d z -> c d w h x y z')
        image = image.reshape(cls_num, z_p*w_p*h_p, *patch_size)

        voxel_mark = rearrange(voxel_mark, 'c (d z) -> c d z',z=patch_size[2])
        voxel_mark = np.repeat(voxel_mark, w_p*h_p, axis=1)

    return image, target, voxel_mark, [z_p,w_p,h_p]

def process_anyres_image(image, target, grid_pinpoints, processor_size, augmentator=None, flag_2D=False):
    """
    Process an image with variable resolutions.

    Args:
        image: The input image to be processed. (H W D)
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        processor_size: input shape of the vision encoder [height, width, depth]
    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # FIXME: determine grid_pinpoints from image sizes.

    # grid_pinpoints 
    # [(m,2n),(2m,n),(2m,2n),(3m,n),(m,3n)]

    possible_resolutions = grid_pinpoints

    best_resolution = select_best_resolution(image.shape, possible_resolutions) # best resolution on h w dimension for now
    
    image_padded, voxel_mark, target_padded, anyres_size = resize_and_pad_image(image, target, best_resolution, flag_2D=flag_2D)

    if augmentator is not None:
        data_dict = {'image': image_padded}
        aug_data_dict = augmentator(data_dict)
        image_padded = aug_data_dict['image']

    patches, targets, voxel_mark, patch_num = divide_to_patches(image_padded, target_padded, voxel_mark, processor_size, flag_2D) # a list of length D, each element shape: N_patch patch_size patch_size

    return patches, targets, voxel_mark, anyres_size, patch_num


def padimgmask_if_necessary(image=None, mask=None,crop_size=(256,256,96)):
    # image size >= crop size 
    if not (image is None):
        c, h, w, d = image.shape
        croph, cropw, cropd = crop_size
        pad_in_h = 0 if h >= croph else croph - h
        pad_in_w = 0 if w >= cropw else cropw - w
        pad_in_d = 0 if d >= cropd else cropd - d
        if pad_in_h + pad_in_w + pad_in_d > 0:
            pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
            image = F.pad(image, pad, 'constant', 0)   # chwd
    
    if not (mask is None):
        n, h, w, d = mask.shape
        croph, cropw, cropd = crop_size
        pad_in_h = 0 if h >= croph else croph - h
        pad_in_w = 0 if w >= cropw else cropw - w
        pad_in_d = 0 if d >= cropd else cropd - d
        if pad_in_h + pad_in_w + pad_in_d > 0:
            pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
            mask = F.pad(mask, pad, 'constant', -1)   # nhwd
    
    return image, mask, d
    
def padpatch_if_necessary(patch,patch_size=(256,256,96),is_mask=False):
    # NOTE: depth must be pad to 96
    
    flag5 = len(list(patch.shape)) == 5

    if flag5:
        b, c, h, w, d = patch.shape
        mark = torch.zeros(b,patch_size[-1])
        mark[:,:d] = 1
        t_h, t_w, t_d = patch_size
        pad_in_h = 0 if h >= t_h else t_h - h
        pad_in_w = 0 if w >= t_w else t_w - w
        pad_in_d = 0 if d >= t_d else t_d - d
    else:
        c, h, w, d = patch.shape
        mark = torch.zeros(patch_size[-1])
        mark[:d] = 1
        t_h, t_w, t_d = patch_size
        pad_in_h = 0 if h >= t_h else t_h - h
        pad_in_w = 0 if w >= t_w else t_w - w
        pad_in_d = 0 if d >= t_d else t_d - d
    
    if pad_in_h + pad_in_w + pad_in_d > 0:
        pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
        if is_mask:
            patch = F.pad(patch, pad, 'constant', -1)
        else:
            patch = F.pad(patch, pad, 'constant', 0)   # chwd

    return patch, mark
    