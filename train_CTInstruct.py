from dataset.qwvl2_dataloader_clip_seg import Qwenvl2_SFTDataset,TrainQwenvl2_ModelCollator, SFT_CombinedDataset
from transformers import Trainer, AutoConfig
from dataset.qwen_processor_clip import Qwen2VLProcessor
from model.llm_vision_feature_generator import LLM_Vision_Encoder
from model.Qwen import Qwen2VLForConditionalGeneration
import os
import torch
import numpy as np
import random

import torch.distributed as dist

from collections import OrderedDict
from config.sft_config_SFT_multi_task_final import qwen2vl_sftconfig

from peft import get_peft_model, LoraConfig, TaskType
from metric.loss import BinaryDice_with_BCE_Loss

from transformers import TrainerCallback

class CustomSaveCallback(TrainerCallback):
    def __init__(self,tokenizer, save_steps):
        self.tokenizer = tokenizer
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            save_path = args.output_dir + f'/checkpoint-{state.global_step}'
            os.makedirs(save_path,exist_ok=True)
            model.save_pretrained(save_path) # safe_serialization=False
            self.tokenizer.save_pretrained(save_path)

            ##### If use lora #####
            # trained_params = {}
            # for name, param in model.named_parameters():
            #     if param.requires_grad and "lora" not in name.lower():
            #         trained_params[name] = param.data.cpu()
            
            # if trained_params:
            #     save_file(trained_params, os.path.join(save_path, "trained_params.safetensors"))
                
            #     with open(os.path.join(save_path, "trained_modules.json"), "w") as f:
            #         json.dump(list(trained_params.keys()), f)


def random_seed(seed=42, rank=0):
    """Seed everything"""
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def load_vision_model(config):
    visual = LLM_Vision_Encoder(config)
    total_params = sum(p.numel() for p in visual.parameters())
    print(f"\nSFT visual parameters: {total_params/1e6:.2f}M")
    if hasattr(config, 'vision_encoder_path'):
        model_dict=torch.load(config.vision_encoder_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            if 'visual_encoder' in k:
                new_state_dict[k.replace('module.visual_encoder','encoder')] = v
            elif "transformer_decoder" in k:
                new_state_dict[k.replace('module.transformer_decoder','transformer_decoder')] = v
            elif 'decoder' in k:
                new_state_dict[k.replace('module.decoder','decoder')] = v
            elif 'avg_pool_ls' in k:
                new_state_dict[k.replace('module.avg_pool_ls','avg_pool_ls')] = v
            elif 'projection_layer' in k:
                new_state_dict[k.replace('module.projection_layer','projection_layer')] = v
            elif 'pos_embedding' in k:
                new_state_dict[k.replace('module.pos_embedding','pos_embedding')] = v
            elif 'mask_embed_proj' in k:
                new_state_dict[k.replace('module.mask_embed_proj','mask_embed_proj')] = v
        
        visual.load_state_dict(new_state_dict, strict=False)

    return visual.bfloat16()

def prepare_dataset(config):
    train_dataset_list=[]
    test_dataset_list=[]
    train_weights = []
    test_weights = []
    for task_name in config.task_list:
        task = config.task_list[task_name]
        train_dataset=Qwenvl2_SFTDataset(task['train'])
        train_dataset_list.append(train_dataset)
        train_weights.append(task['w'])
        if 'test' in task:
            test_dataset=Qwenvl2_SFTDataset(task['test'])
            test_dataset_list.append(test_dataset)
            test_weights.append(task['w'])

    return SFT_CombinedDataset(train_dataset_list, train_weights),SFT_CombinedDataset(test_dataset_list, test_weights)

config=qwen2vl_sftconfig()
training_args = config.training_args

print("Initialze Processor")
processor = Qwen2VLProcessor.from_pretrained(
    config.cache_dir, # config.llm_backbone
    cache_dir=config.cache_dir)
processor.crop_size = config.crop_size

print("Initialize Processor tokenizer")
processor.tokenizer.add_tokens(['<SEG>'])
processor.tokenizer.add_tokens(['<binary_answer>'])
processor.tokenizer.add_tokens(['<multiple_choice>'])
processor.tokenizer.add_tokens(['<report_generation>'])
processor.tokenizer.add_tokens(['<segmentation>'])
processor.tokenizer.add_tokens(['<diagnosis>'])
config.seg_token_idx = processor.tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
print('seg_token_idx',config.seg_token_idx)
processor.tokenizer.add_tokens(['<disease_confidence>'])
config.disease_conf_token_idx = processor.tokenizer("<disease_confidence>", add_special_tokens=False).input_ids[0]
print('disease_conf_token_idx',config.disease_conf_token_idx)
tokenizer = processor.tokenizer

print("Initialize Dataset")
train_dataset,test_dataset=prepare_dataset(config)
print("train",len(train_dataset),"test",len(test_dataset))

print("Initialize Data Collator")
data_collator=TrainQwenvl2_ModelCollator(processor=processor, crop_size=config.crop_size, patch_max=config.patch_max)


print("Initialize Model")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.llm_backbone,
    cache_dir=config.cache_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
    )


model.seg_token_idx = config.seg_token_idx
model.disease_conf_token_idx = config.disease_conf_token_idx
model.dice_loss = BinaryDice_with_BCE_Loss()

model.visual = load_vision_model(config) 
model.visual = model.visual.to(dist.get_rank())
model.freeze_vision = config.freeze_vision
# If add any special tokens, do this
model.resize_token_embeddings(len(tokenizer))

if config.llm_lora:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
        )
    model=get_peft_model(model, lora_config)
    print("**************** using lora *************")
                     
random_seed(seed=42, rank=dist.get_rank())

for p in model.visual.parameters():
    p.requires_grad = True

for name,param in model.named_parameters():
    if 'rad_class_heads' in name or 'ctrate_class_heads' in name:
        param.requires_grad = True

os.makedirs(config.save_path,exist_ok=True)

print(f"Model dtype: {next(model.visual.encoder.parameters()).dtype}")
model.visual = model.visual.to(torch.bfloat16)
print(f"Model dtype2: {next(model.visual.encoder.parameters()).dtype}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    callbacks=[CustomSaveCallback(processor.tokenizer,config.save_steps)]
)

model.train()

trainer.train()
