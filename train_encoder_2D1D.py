from trainer.trainer_ablation import trainer_ablation
from datamanager.datamanager_ablation import data_Manager
import torch
import warnings
from utils import load_config,save_tensorboard, get_accumulation_steps,print_model_parameters
from metric.get_metric import get_criterion
import deepspeed
from deepspeed.comm import reduce, get_rank, get_world_size
from model.encoder_ablation import Encoder_ablation
import os
import torch.distributed as dist

training_config_path = "config/train_ablation_seg_cls_image_text.yaml"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

training_config = load_config(training_config_path)
tensorboard_path=training_config['tensorboard_path']

if 'criterion_dict' in training_config:
    criterion_dict=get_criterion(criterion_dict=training_config["criterion_dict"])
else:
    criterion_dict = {}


flag_2D_dict = {i:False for i in training_config["task_list"]}
if 'flag_2D' in training_config:
    for tn in training_config['flag_2D']:
        flag_2D_dict[tn]=True

os.makedirs(training_config['savedir'],exist_ok=True)

label_dict = training_config["label_dict"] if 'label_dict' in training_config else {}
label_description = training_config['label_description'] if 'label_description' in training_config else {}

model = Encoder_ablation(config=training_config,
    label_dict = label_dict,
    label_description = label_description,
    task_names= training_config["task_list"]
)

print_model_parameters(model, model_name="Encoder")

if 'vision_checkpoint' in training_config:

    model_keys = list(model.state_dict().keys())

    model_dict=torch.load(training_config['vision_checkpoint'], map_location="cpu",weights_only=True)
    new_state_dict = {}
    for k, v in model_dict.items():
        new_state_dict[k.replace('module.','')] = v
    model.load_state_dict(new_state_dict)

multi_task_grad_interval = training_config['multi_task_grad_interval'] if 'multi_task_grad_interval' in training_config else None
accumulate_grad_interval = get_accumulation_steps(get_world_size(), training_config['batchsize'], training_config['accumulate_batchsize']) if 'accumulate_batchsize' in training_config else None

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,config_params=training_config["deepspeed_config"])

model.text_encoder.to_device(dist.get_rank())

if get_rank()==0:
    writer = save_tensorboard(tensorboard_path)
else:
    writer=None

data_manager=data_Manager(
    task_names=training_config["task_list"],
    batchsize=training_config["batchsize"],
    crop_size=training_config["crop_size"],
    meta_data_dic=training_config["meta_data_dict"],
    patch_max=training_config['patch_max'],
    flag_2D_dic=flag_2D_dict
)

trainer=trainer_ablation(
    task_list = training_config["task_list"],
    data_manager=data_manager,
    model=model_engine,
    optimizer=optimizer,
    criterion_dict=criterion_dict,
    sampleweight_dict=training_config["sampleweight_dict"],
    flag_2D_dict=flag_2D_dict,
    savedir=training_config["savedir"],
    writer=writer,
    ckpt=training_config["ckpt"],
    resume_from_step=training_config["resume_from_step"],
    freeze_last_layer = training_config['freeze_last_layer'],
    accumulate_grad_interval = accumulate_grad_interval,
    multi_task_grad_interval = multi_task_grad_interval
    )

criterion_dict = training_config["criterion_dict"] if 'criterion_dict' in training_config else {}
trainer.train(criterion_dict=criterion_dict,
            total_iters=training_config["max_iters"],
            eval_interval=training_config["eval_interval"],
            eval_iter_per_task = training_config["eval_iter_per_task"]
            )
if get_rank()==0:
    writer.close()