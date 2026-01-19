      
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
from .loss import *
import nibabel as nib
import os
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, recall_score

def get_identifier(image_path_list):
    assert len(image_path_list) == 1 and len(image_path_list[0]) == 1
    dataset = image_path_list[0][0].split('/')[-3]
    index = image_path_list[0][0].split('/')[-1].split('.')[0]
    return '_'.join([dataset,index])

def to_original_size(pred_masks, masks, patch_cords, image_size):
    dic = {}
    pred_mask_original_size = []
    mask_original_size = []
    for b in range(len(pred_masks)):
        pred_mask_original_size.append([])
        mask_original_size.append([])
        for nt in range(len(pred_masks[b])):
            size = image_size[b][nt] # H W D
            pred_original = torch.zeros(size)
            mask_original = torch.zeros(size)
            
            for p in range(len(pred_masks[b][nt])):
                pred_patches = pred_masks[b][nt][p][0] # 256 256 32
                mask_patches = masks[b][nt][p][0] # 256 256 32
                h_s, h_e, w_s, w_e, d_s, d_e = patch_cords[b][nt][p] #
                original_h, original_w, original_d = h_e-h_s, w_e-w_s, d_e-d_s
                patch_h, patch_w, patch_d = pred_patches.shape
                if patch_h > original_h or patch_w > original_w or patch_d > original_d:
                    pred_original[h_s:h_e, w_s:w_e, d_s:d_e] = pred_patches[:original_h,:original_w, :original_d]
                    mask_original[h_s:h_e, w_s:w_e, d_s:d_e] = mask_patches[:original_h,:original_w, :original_d]
                    
                else:
                    pred_original[h_s:h_e, w_s:w_e, d_s:d_e] = pred_patches
                    mask_original[h_s:h_e, w_s:w_e, d_s:d_e] = mask_patches 
            pred_mask_original_size[-1].append(pred_original)
            mask_original_size[-1].append(mask_original)
            

    return pred_mask_original_size, mask_original_size

def get_binary_mask_from_logits(logits):
    predict = torch.sigmoid(logits)
    predict = torch.where(predict>0.5, torch.ones_like(predict),torch.zeros_like(predict))
    return predict

def cal_dice(pred_masks, gt_masks):
    total_dice = []
    for b in range(len(pred_masks)):
  
        pred_origin = get_binary_mask_from_logits(pred_masks[b][0])

        gt_origin = gt_masks[b][0][0]
        
        gt_origin[gt_origin==-1] = 0
        
        predict = pred_origin[0][0]
        
        target = gt_origin.to(pred_origin.device)

        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + 1e-7
        union = torch.sum(predict + target, dim=1) + 1e-7

        dice_coefficient = intersection / union
        total_dice.append(dice_coefficient.cpu().numpy())

    return np.mean(total_dice)

def get_pad_start_end(padded_tensor, pad_value=-1):

    x_mask = (padded_tensor != pad_value).any(dim=(1, 2)).float()
    x_start = torch.argmax(x_mask, axis=0)
    x_end = x_mask.shape[0] - torch.argmax(x_mask.flip(0), axis=0)

    y_mask = (padded_tensor != pad_value).any(dim=(0, 2)).float()
    y_start = torch.argmax(y_mask, axis=0)
    y_end = y_mask.shape[0] - torch.argmax(y_mask.flip(0), axis=0)

    z_mask = (padded_tensor != pad_value).any(dim=(0, 1)).float()
    z_start = torch.argmax(z_mask, axis=0)
    z_end = z_mask.shape[0] - torch.argmax(z_mask.flip(0), axis=0)

    return x_start, x_end, y_start, y_end, z_start, z_end


def write_image(seg_root_dir, images, pred_masks, gt_masks, patch_num, identifier):
    pred_out_path = os.path.join(seg_root_dir,identifier+'_pred.nii.gz')
    gt_out_path = os.path.join(seg_root_dir,identifier+'_gt.nii.gz')
    image_out_path = os.path.join(seg_root_dir,identifier+'_image.nii.gz')
    pred_masks = pred_masks.squeeze()
    gt_masks = gt_masks.squeeze()
    if len(images.shape)>4:
        images = images.squeeze()
    if len(images.shape) == 3:
        images = images[None,:]

    pred_masks = get_binary_mask_from_logits(pred_masks)
    
    images = rearrange(images, '(z x y) h w d -> (x h) (y w) (z d)', z = patch_num[0],x=patch_num[1],y=patch_num[2])
    
    x_start, x_end, y_start, y_end, z_start, z_end = get_pad_start_end(gt_masks)
    
    gt_origin = gt_masks[x_start:x_end, y_start:y_end, z_start:z_end].detach().cpu().numpy().astype(np.uint8)
    pred_origin = pred_masks[x_start:x_end, y_start:y_end, z_start:z_end].detach().cpu().to(torch.float32).numpy().astype(np.uint8)
    image_origin = images[x_start:x_end, y_start:y_end, z_start:z_end].detach().cpu().to(torch.float32).numpy()
    

    nib.save(nib.Nifti1Image(pred_origin, np.eye(4)), pred_out_path)
    nib.save(nib.Nifti1Image(gt_origin, np.eye(4)), gt_out_path)
    nib.save(nib.Nifti1Image(image_origin, np.eye(4)), image_out_path)
    return pred_out_path, gt_out_path, image_out_path

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # PyTorch 1.10+

class StableCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()        
        self.loss = nn.CrossEntropyLoss()
    def check_nan(self, output):
        res = []
        output=torch.exp(output)
        for idx in range(output.shape[0]):
            if torch.isnan(torch.exp(output[idx])).sum()>0:
                res.append(idx)
        return res
                
    def forward(self, logits, targets):
        res1 = self.check_nan(logits)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        res2 = self.check_nan(logits)
        loss =  self.loss(logits, targets)
        return loss

def get_criterion(criterion_dict):

    for task_name in criterion_dict:
        if criterion_dict[task_name] in ["BinaryDice_with_BCE"]:
            criterion_dict[task_name]=BinaryDice_with_BCE_Loss()
        elif criterion_dict[task_name] in ["CE"]:
            criterion_dict[task_name]=nn.CrossEntropyLoss()
        elif criterion_dict[task_name] in ["Stable_CE"]:
            criterion_dict[task_name] = StableCrossEntropy()
        elif criterion_dict[task_name] in ["BCE"]:
            criterion_dict[task_name]=nn.BCEWithLogitsLoss()
        elif criterion_dict[task_name] in ["BinaryDice_with_Focal_Loss"]:
            criterion_dict[task_name]=BinaryDice_with_Focal_Loss()
        else:
            raise ValueError("Not supported loss!")
    return criterion_dict

def get_diagnosis_metric_bce(y_true, y_probs):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    aucs,accs,precisions,f1s,threshs = [],[],[],[],[]

    for d in range(len(y_true[0])):
        y_prob_d = y_probs[:,d]
        y_true_d = y_true[:,d]
        try:
            auc = roc_auc_score(y_true_d, y_prob_d)
        except:
            auc = 0
        max_f1,best_thresh = 0,0
        for threshold in np.arange(0,1,0.001):
            y_pred_thresh = (y_prob_d >= threshold).astype(int)
            f1 = f1_score(y_true_d, y_pred_thresh)
            if f1>max_f1:
                max_f1 = f1
                best_thresh = threshold
        y_pred_d = (y_prob_d >= best_thresh).astype(int)
        acc = accuracy_score(y_true_d, y_pred_d)
        precision = precision_score(y_true_d, y_pred_d)
        accs.append(acc)
        f1s.append(max_f1)
        precisions.append(precision)
        threshs.append(best_thresh)
        aucs.append(auc)
    return aucs,accs,precisions,f1s,threshs


def cal_diagnosis_metric(y_true, y_pred):

    recall = recall_score(y_true, y_pred)
    
    # Precision
    precision = precision_score(y_true, y_pred)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)

    return recall, precision, acc, f1