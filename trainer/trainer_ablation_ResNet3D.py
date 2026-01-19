from tqdm import tqdm
import random
import torch
from torch.cuda.amp import autocast as autocast
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score
from deepspeed.comm import get_rank
import deepspeed.comm as dist
from time import time
import torch.nn.functional as F
import torch.nn as nn

def dice_metric(preds, targets, mark,epsilon=1e-6):

    class_dice=[]
    for pred,target in zip(preds,targets):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        class_dice.append(dice)
    return np.array(class_dice)

def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def transfer_task_name(task_name):
    if "segmentation" in task_name:
        return "segmentation"
    elif "classification" in task_name or "treatment_planning" in task_name:
        return "classification"    
    elif 'image_text' in task_name:
        return "image_text"
    elif 'dino' in task_name:
        return "dino"

class trainer_ablation():
    def __init__(self,
        task_list,
        data_manager, 
        model, 
        optimizer,
        criterion_dict,
        sampleweight_dict,
        flag_2D_dict,
        savedir="./save",
        writer=None,
        ckpt=None,
        resume_from_step=0,
        freeze_last_layer=0, 
        accumulate_grad_interval = None,
        multi_task_grad_interval = None
    ):
        self.data_manager=data_manager
        self.model=model
        self.start_iters = resume_from_step
        self.task_list = task_list

        self.task_to_idx = {name: idx for idx, name in enumerate(self.task_list)}
        self.idx_to_task = {idx: name for idx, name in enumerate(self.task_list)}

        if ckpt is not None:
            print("loading checkpoint from:", ckpt)
            self.model.load_checkpoint(ckpt)

        self.optimizer=optimizer
        self.criterion_dict=criterion_dict
        self.writer=writer
        self.num_iters=0
        self.local_rank=get_rank()
        self.sampleweight_dict=sampleweight_dict
        self.flag_2D_dict=flag_2D_dict
        self.savemodelonly=True
        self.savedir=savedir
        
        self.weights=[sampleweight_dict[name] for name in self.task_list]
        self.device = torch.device(f'cuda:{self.local_rank}')

        self.world_size = dist.get_world_size()
        self.freeze_last_layer = freeze_last_layer

        self.accumulation_steps = 0
        self.accumulate_grad_interval = accumulate_grad_interval
        self.multi_task_grad_interval = multi_task_grad_interval

    def train_one_iter(self,task_name,criterion=None,it=0):
        
        try:
            batch = next(self.data_manager.trainiter_dict[task_name])
        except StopIteration:
            self.data_manager.trainiter_dict[task_name]=iter(self.data_manager.trainloader_dict[task_name])
            batch = next(self.data_manager.trainiter_dict[task_name])
    
        self.model.train()

        if "segmentation" in task_name:
            image = batch["image"]
            patch_num = batch['patch_num']
            masks = batch['mask']
            chosen_label_idx = batch['chosen_label_idx']
            total_bce_loss = torch.tensor(0.0, device=self.model.local_rank)
            total_dice_loss = torch.tensor(0.0, device=self.model.local_rank)
            
            with autocast(dtype=torch.float16):
                seg_logits=self.model.segmentation_forward(image, patch_num, task_name, chosen_label_idx, self.model.local_rank) # b cls_num H W D
            
            num_samples = len(seg_logits)

            for logits,mask in zip(seg_logits, masks):

                cur_bce_loss, cur_dice_loss=criterion(logits,mask[None,:].to(self.model.device))

                total_bce_loss = total_bce_loss + cur_bce_loss
                total_dice_loss = total_dice_loss + cur_dice_loss

            avg_bce_loss = total_bce_loss / num_samples
            avg_dice_loss = total_dice_loss / num_samples
            loss = avg_bce_loss + avg_dice_loss

            metric_names=["bce_loss","dice_loss","total_loss"]
            metric_array=[avg_bce_loss,avg_dice_loss,loss]
        
        elif "classification" in task_name or "treatment_planning" in task_name:
            image=batch["image"]
            if torch.is_tensor(batch["cls_label"][0]):
                cls_label = torch.stack(batch["cls_label"]).to(self.model.local_rank)
            else:
                cls_label=torch.tensor(batch["cls_label"]).to(self.model.local_rank)
            
            with autocast(dtype=torch.float16):
                cls_logits, cls_label = self.model.classification_forwad(image, cls_label,task_name, self.model.local_rank,self.flag_2D_dict[task_name])
            

            if isinstance(criterion, nn.CrossEntropyLoss):
                loss=criterion(cls_logits,cls_label)
            else:
                loss=criterion(cls_logits,cls_label.float())
            metric_names=["ce_loss"]
            metric_array=torch.tensor([loss.item()])
        
        elif 'image_text' in task_name:
            torch.cuda.empty_cache()
            image = batch["image"]
            text = batch["text"]
            with autocast(dtype=torch.float16):
                loss = self.model.imagetext_forward(image, text, self.model.local_rank)
            metric_names = ['ita_loss']
            metric_array = torch.tensor([loss.item()])
        
        self.accumulation_steps += 1

        torch.cuda.empty_cache()
        self.model.backward(loss)
        # accumulate grad or bp and update the model

        if self.multi_task_grad_interval is not None:
            accumulate_grad_interval = self.multi_task_grad_interval
        else:
            accumulate_grad_interval = self.accumulate_grad_interval[transfer_task_name(task_name)]
        
        if self.accumulation_steps % accumulate_grad_interval == 0:
            
            self.accumulation_steps = 0

            self.model.step()
        
        return metric_names,metric_array
        
    def train(self,criterion_dict,total_iters,eval_interval,eval_iter_per_task):

        for idx in tqdm(range(self.start_iters, total_iters),disable=self.model.local_rank != 0):

            selected_task_train_idx = torch.zeros(1, dtype=torch.long, device=self.device)

            if self.local_rank==0:
                if self.weights is None:
                    task_name=random.choice(self.task_list)
                else:
                    task_name=random.choices(self.task_list, weights=self.weights, k=1)[0]
                selected_task_train_idx[0] = self.task_to_idx[task_name]
            else:
                task_name = None

            dist.broadcast(selected_task_train_idx, src=0)
            selected_task_train_idx = selected_task_train_idx.item()
            task_name = self.idx_to_task[selected_task_train_idx]

            criterion = criterion_dict[task_name] if task_name in criterion_dict else None
            
            metric_names,metric_array=self.train_one_iter(task_name=task_name,criterion=criterion,it = idx)

            if 'segmentation' in task_name:
                for tensor in metric_array:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 求和操作
                    tensor.div_(self.world_size)

            self.num_iters+=1

            if self.local_rank==0:
               for idx in range(len(metric_names)):
                    loss_name = metric_names[idx]
                    loss_num = metric_array[idx]
                    self.writer.add_scalar(task_name+'/train/'+loss_name, loss_num, self.num_iters)

            if self.num_iters%eval_interval==0:
                with autocast(dtype=torch.float16):
                    self.evaluate(criterion_dict, eval_iter_per_task)
                torch.save(self.model.state_dict(),self.savedir+"/iters_"+str(self.num_iters)+".pth")
            if self.num_iters % (eval_interval//2) == 0:
                torch.save(self.model.state_dict(),self.savedir+"/iters_"+str(self.num_iters)+".pth")

    def eval_one_iter(self,task_name,criterion):

        try:
            batch = next(self.data_manager.testiter_dict[task_name])
        except StopIteration:
            self.data_manager.testiter_dict[task_name]=iter(self.data_manager.testloader_dict[task_name])
            batch = next(self.data_manager.testiter_dict[task_name])
    
        self.model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            if "segmentation" in task_name:
                image = batch["image"]
                masks = batch["mask"]
                patch_num = batch['patch_num']
                chosen_label_idx = batch['chosen_label_idx']

                total_bce_loss = torch.tensor(0.0, device=self.model.local_rank)
                total_dice_loss = torch.tensor(0.0, device=self.model.local_rank)
                with autocast(dtype=torch.float16):
                    seg_logits = self.model.segmentation_forward(image, patch_num, task_name, chosen_label_idx, self.model.local_rank) # b cls_num H W D
                num_samples = len(seg_logits)

                for logits,mask in zip(seg_logits, masks):
                    cur_bce_loss, cur_dice_loss=criterion(logits,mask[None,:].to(self.model.device))
                    
                    total_bce_loss = total_bce_loss + cur_bce_loss
                    total_dice_loss = total_dice_loss + cur_dice_loss

                avg_bce_loss = total_bce_loss / num_samples
                avg_dice_loss = total_dice_loss / num_samples
                loss = avg_bce_loss + avg_dice_loss

                metric_names=["bce_loss","dice_loss","total_loss"]
                metric_array=[avg_bce_loss,avg_dice_loss,loss]

                metric = None

            elif "classification" in task_name or "treatment_planning" in task_name:

                image=batch["image"]

                if torch.is_tensor(batch["cls_label"][0]):
                    cls_label = torch.stack(batch["cls_label"]).to(self.model.local_rank)
                else:
                    cls_label=torch.tensor(batch["cls_label"]).to(self.model.local_rank)
                
                with autocast(dtype=torch.float16):
                    cls_logits, cls_label = self.model.classification_forwad(image,cls_label,task_name,self.model.local_rank, self.flag_2D_dict[task_name])

                if isinstance(criterion, nn.CrossEntropyLoss):
                    bce_flag = False
                    loss=criterion(cls_logits,cls_label)
                else:
                    bce_flag=True
                    loss=criterion(cls_logits,cls_label.float())
                metric_array=[loss.item()]

                if bce_flag:
                    metric_names = ["bce_loss"]
                    pred = torch.sigmoid(cls_logits)
                    total_accs = 0
                    thresholds = 0
                    cls_label = cls_label.cpu().detach().numpy()
                    pred = pred.cpu().detach().numpy()
                    for c in range(cls_label.shape[1]):
                        best_metric = 0
                        best_threshold = 0
                        cur_label = cls_label[:,c]
                        for threshold in np.arange(0,1,0.01):
                            cur_pred = (pred[:,c] >= threshold)
                            metric = accuracy_score(cur_label, cur_pred)
                            if metric > best_metric:
                                best_threshold = threshold
                                best_metric = metric
                        total_accs += best_metric
                        thresholds += best_threshold
                    metric_names.append('threshold_avg')
                    metric_array.append(thresholds/cls_label.shape[1])
                    metric = total_accs / cls_label.shape[1]
                else:
                    metric_names=["ce_loss"]
                    probability = F.softmax(cls_logits, dim=1)
                    _, pred = torch.max(probability, dim=1)
                    metric = accuracy_score(cls_label.cpu().detach().numpy(),pred.cpu().detach().numpy())
                
                metric_names=["ce_loss"]
                metric_array=torch.tensor([loss.item()])
            
            elif 'image_text' in task_name:

                image = batch["image"]
                text = batch["text"]
                loss = self.model.imagetext_forward(image, text, self.model.local_rank)
                metric_names = ['ita_loss']
                metric_array = torch.tensor([loss.item()])
                metric = None
        
        return metric_names,metric_array,metric
    
    def evaluate(self, criterion_dict, eval_iter_per_task):

        torch.cuda.empty_cache()

        for task_name in self.task_list:

            total_loss = []
            if 'segmentation' in task_name:
                test_iter_per_task = eval_iter_per_task['segmentation']
            elif 'image_text' in task_name:
                test_iter_per_task = eval_iter_per_task['image_text']
            else:
                test_iter_per_task = eval_iter_per_task['classification']
            for _ in tqdm(range(test_iter_per_task)):

                criterion = criterion_dict[task_name] if task_name in criterion_dict else None
                metric_names, metric_array, metric = self.eval_one_iter(task_name=task_name,criterion=criterion)

                if 'segmentation' in task_name:
                    for tensor in metric_array:
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        tensor.div_(self.world_size)

                for index in range(len(metric_array)):
                    if len(total_loss) == len(metric_array):
                        total_loss[index] += metric_array[index]
                    else:
                        total_loss.append(metric_array[index])
            
            if self.local_rank==0:
                for idx in range(len(metric_names)):
                    loss_name = metric_names[idx]
                    loss_num = total_loss[idx] / test_iter_per_task
                    self.writer.add_scalar(task_name+'/eval/'+loss_name, loss_num, self.num_iters)
                
                if metric is not None:
                    self.writer.add_scalar(task_name+'/eval/metric', metric, self.num_iters)