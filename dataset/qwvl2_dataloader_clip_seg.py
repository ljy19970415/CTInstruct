import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import math
from .utils import *
import random
import os

class SFT_CombinedDataset(Dataset):
    def __init__(self, datasets, sampling_probs):
        self.datasets = datasets
        self.sampling_probs = sampling_probs
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = torch.multinomial(torch.tensor(self.sampling_probs), 1).item()
        sample_idx = torch.randint(0, self.lengths[dataset_idx], (1,)).item()
        return self.datasets[dataset_idx][sample_idx]

class Qwenvl2_SFTDataset(Dataset):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.samples=self.build_dataset(datapath)
        
    def build_dataset(self, datapath: str):
        with open(datapath, 'r') as f:
            samples = json.load(f)
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample=self.samples[index]
        question_list=sample["question"]
        answer=sample["answer"]
        
        image_path_list=sample["image_path"]

        if 'mask_path' in sample:
            mask_path_list = sample['mask_path']
            choose_label_list = sample['choose_labels']
            renorm_box_list = sample['renorm_y1x1z1_y2x2z2']
            renorm_flag = True
        else:
            mask_path_list = None
            choose_label_list = None
            renorm_box_list = None
            renorm_flag = False
        
        if 'targets' in sample:
            target = sample['targets'] # [1,0,1,...,0]
        else:
            target = None

        return question_list, answer, image_path_list, mask_path_list, choose_label_list, renorm_box_list, renorm_flag, target

def Qwenvl2_build_qaimage(processor, grids, crop_size, feature):
    # question_list, answer, image_path_list, renorm_flag
    question_list, answer, image_path_list, mask_path_list, choose_label_list, renorm_box_list, renorm_flag, target = feature
    messages = [
        {"role": "user", "content": question_list},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,add_special_tokens=False,
    )
    # prompt (do not contain answer)
    # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> ..<|im_end|>\n<|im_start|>assistant\n
    if isinstance(image_path_list,str):
        image_path_list = [image_path_list]
    image_list=[]
    mark_list = []
    patch_num_list = []

    mask_list = []
    image_size_list = []
    patch_cords_list = []

    for idx,img_path in enumerate(image_path_list):
        # image p 256 256 32
        # mark p 32
        flag_2D = check_2D(img_path)
        
        if 'RAD-chestCT' in img_path or 'CT-RATE' in img_path:
            image = load_datum(img_path, flag_2D=False)
        else:
            if "npz" in img_path:
                image=np.load(img_path)['arr_0']
            else:
                if os.path.exists(img_path):
                    image=np.load(img_path)
                else:
                    npz_path = img_path[:-4]+'.npz'
                    image=np.load(npz_path)['arr_0']

        if mask_path_list is not None:
            mask_path, choose_label, renorm_box = mask_path_list[idx], choose_label_list[idx], renorm_box_list[idx]
            mask = load_mask(mask_path, image.shape, choose_label, renorm_box)
        else:
            mask = None

        image, mask, mark, padded_size, patch_num = process_anyres_image(image, mask, grids, crop_size)
        
        image_list.append(image)
        mark_list.append(mark)
        patch_num_list.append(patch_num)
        if mask is not None:
            mask_list.append(mask)

    
    inputs = processor(text=prompt, images=image_list, patch_nums=patch_num_list, return_tensors="pt",add_special_tokens=True)  # .to(0, torch.float16)
      
    answer_input_ids = processor.tokenizer.encode(
        answer,
        return_tensors="pt",
        max_length=4096,
        add_special_tokens=False,
        padding='longest',
        truncation=True,
    )

    return dict(
        q_input_ids=inputs["input_ids"],
        pixel_values= inputs["pixel_values"],
        a_input_ids=answer_input_ids,
        marks=mark_list,
        masks=mask_list,
        image_grid_thw = inputs['image_grid_thw'],
        patch_num = patch_num_list,
        image_path_list=image_path_list,
        choose_label_list=choose_label_list
    )
    #'''

class TrainQwenvl2_ModelCollator:
    def __init__(self, processor, crop_size, patch_max, IGNORE_INDEX=-100):
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX
        self.crop_size = crop_size
        self.grids = gen_grid_points(patch_max,crop_size)

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []
        image_grid_thw = []
        marks_list = []
        patch_num_list = []
        masks_list = []

        answer_texts = []
        targets = []
        for feature in features:
            answer_texts.append(feature[1])
            # question_list, answer, image_path_list, mask_path_list, choose_label_list, renorm_box_list, renorm_flag
            qaimage_output = Qwenvl2_build_qaimage(
                self.processor, self.grids, self.crop_size, feature
            )
            # concat question and answer
            # temp_input_ids 1, question_answer_sequence_length
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output["q_input_ids"], qaimage_output["a_input_ids"]
            )
            marks_list.append(qaimage_output['marks'])
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output["pixel_values"])
            image_grid_thw.append(qaimage_output["image_grid_thw"])
            patch_num_list.append(qaimage_output['patch_num'])
            masks_list.append(qaimage_output["masks"])
            if feature[-1] is not None:
                targets.append(torch.tensor(feature[-1]))
            else:
                targets.append(torch.tensor([0]))
            

        max_input_len = max(max_input_len_list)

        # pad all input_ids to the longest input ids within the batch
        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        # concat all the labels to the longest input_ids within the batch
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )

        # final_pixel_values = torch.concat(pixel_values, axis=0) # [pixel_number within a batch, vision_dim]
        final_pixel_values = pixel_values
        image_grid_thw=torch.concat(image_grid_thw,axis=0) # [number of images within a batch, 3]
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
            "image_grid_thw":image_grid_thw,
            "marks":marks_list,
            "patch_num":patch_num_list,
            "masks":masks_list,
            'targets':torch.concat(targets)
        }

    

class Imageonly_Collator:
    def __init__(self, crop_size, batch_size):
        self.crop_size = crop_size
        self.grids = gen_grid_points(8,crop_size)

    def load_qaimage(self, grids, crop_size, feature):
        # question_list, answer, image_path_list, renorm_flag
        question_list, answer, image_path_list, mask_path_list, choose_label_list, renorm_box_list, renorm_flag, target = feature
        
        if isinstance(image_path_list,str):
            image_path_list = [image_path_list]
        image_list=[]
        mark_list = []
        patch_num_list = []
        mask_list = []
        image_size_list = []

        for idx,img_path in enumerate(image_path_list):
            # image p 256 256 32
            # mark p 32
            flag_2D = check_2D(img_path)
        
            if 'SAM/processed_files_v4' in img_path:
                if "npz" in img_path:
                    image=torch.tensor(np.load(img_path)['arr_0'])
                else:
                    image=torch.tensor(np.load(img_path))
            elif 's3://' in img_path:
                image = load_from_bucket(img_path)
            else:
                image = load_datum(img_path,flag_2D=flag_2D)

            if mask_path_list is not None:
                mask_path, choose_label, renorm_box = mask_path_list[idx], choose_label_list[idx], renorm_box_list[idx]
                mask = load_mask(mask_path, image.shape, choose_label, renorm_box)
            else:
                mask = None

            image, mask, mark, padded_size, patch_num = process_anyres_image(image, mask, grids, crop_size) # self.augmentator

            image_list.append(image)
            mark_list.append(mark)
            patch_num_list.append(patch_num)
            if mask is not None:
                mask_list.append(mask)
        
        return dict(
            pixel_values= image_list,
            marks=mark_list,
            masks=mask_list,
            patch_num = patch_num_list,
            image_path_list=image_path_list,
            choose_label_list=choose_label_list
        )
        #'''

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        pixel_values = []
        marks_list = []
        patch_num_list = []
        image_path_list = []
        question_list = []
        masks_list = []
        targets = []
        choose_label_list = []

        #print(len(features))
        dic = {}
        dic['B'] = len(features)
        cnt = 0
        answer_texts = []
        for feature in features:
            # question_list, answer, image_path_list, renorm_flag
            answer_texts.append(feature[1])
            question_list.append(feature[0])
            qaimage_output = self.load_qaimage(
                self.grids, self.crop_size, feature
            )
            marks_list.append(qaimage_output['marks'])
            patch_num_list.append(qaimage_output['patch_num'])
            pixel_values.append(qaimage_output["pixel_values"])
            image_path_list.append(qaimage_output['image_path_list'])
            if qaimage_output['choose_label_list'] is not None:
                choose_label_list.append(qaimage_output['choose_label_list'])
            else:
                choose_label_list.append([''])
            masks_list.append(qaimage_output['masks'])
            if feature[-1] is not None:
                targets.append(torch.tensor(feature[-1]))
            else:
                targets.append(torch.tensor([0]))

        return {
            "patch_num":patch_num_list,
            "pixel_values": pixel_values,
            "marks": marks_list,
            "answer": answer_texts,
            "image_path_list":image_path_list,
            'question':question_list,
            "masks":masks_list,
            'targets':torch.concat(targets),
            'choose_label_list':choose_label_list
        }

