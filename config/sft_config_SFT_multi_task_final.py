from transformers import TrainingArguments
from pathlib import Path
import os


def _find_project_root(start: Path) -> Path:
    markers = [".git", "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt"]
    for p in [start] + list(start.parents):
        if any((p / m).exists() for m in markers):
            return p
    return start.parent

class qwen2vl_sftconfig:
    def __init__(self):
        env_root = os.getenv("PROJECT_ROOT")
        if env_root:
            self.base_dir = Path(env_root).resolve()
        else:
            self.base_dir = _find_project_root(Path(__file__).resolve())

        def p(*parts):
            return str(self.base_dir.joinpath(*parts))

        self.cache_dir = p(".cache")

        self.name = "ctinstruct"

        self.vision_encoder_path = p("checkpoint", "vision_encoder.pth")
        self.encoder_config_path = p("config", "train_ablation_seg_cls_image_text.yaml")

        self.BATCH_SIZE = 64
        self.patch_max = 6

        self.freeze_vision = False
        self.llm_lora = False

        self.llm_backbone = "Qwen/Qwen2-VL-7B-Instruct"
        self.gradient_checkpointing = True

        self.num_train_epochs = 2
        self.num_train_iters = 10000
        self.save_steps = 2500

        self.save_strategy = "no"
        self.save_only_model = True

        self.save_path = p("checkpoint", self.name)

        self.evaluation_strategy="steps"
        self.eval_on_start=False
        self.eval_steps=200000

        self.output_dir=p("checkpoint", self.name)

        # Set to "wandb" if you are using Weights and Biases for logging
        self.logging_dir=p("tensorboard", self.name)

        self.report_to="tensorboard"
        self.visual_first=False
        self.logging_steps=10
        self.logging_first_step=True

        self.dataloader_num_workers=0

        self.per_device_train_batch_size=1
        self.per_device_eval_batch_size=1

        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // self.per_device_train_batch_size

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            self.GRADIENT_ACCUMULATION_STEPS = self.GRADIENT_ACCUMULATION_STEPS // world_size

        self.learning_rate=2e-5
        self.weight_decay=0.0
        self.warmup_ratio=0.1
        self.max_grad_norm=1.0
        self.lr_scheduler_type="cosine" 
        self.deepspeed_config=str(p("config", "zero2.json"))

        self.fp16=False
        self.bf16=True
        self.ddp_find_unused_parameters=False

        
        self.task_list = {
            # 'M3D_Caption':{
            #     'train':p("data","instruction_sft",VQA_M3D_Cap_train.json"),
            #     'test':p("data","instruction_sft","VQA_M3D_Cap_val.json"),
            #     'test_file': p("data","instruction_sft","VQA_M3D_Cap_test.json"),
            #     'w':0.3,
            #     "task_type":"text"
            # },
            # 'RadChest_Binary_Diagnosis':{
            #     'train':p("data","instruction_sft","VQA_RadChestCT_Binary_Diagnosis_train.json"),
            #     'test':p("data","instruction_sft","VQA_RadChestCT_Binary_Diagnosis_test.json"),
            #     'test_file':p("data","instruction_sft","VQA_RadChestCT_Binary_Diagnosis_test.json"),
            #     'w':0.1,
            #     "task_type":"diagnosis"
            #     # "outputpath":os.path.join(self.test_output_dir,"radchest_binary_diagnosis.json")
            # },
            # 'CTRATE_Diagnosis':{# 24128
            #     'train':p("data","instruction_sft","VQA_CTRATE_diagnosis_train.json"),
            #     'test':p("data","instruction_sft","VQA_CTRATE_diagnosis_test.json"),
            #     'test_file':p("data","instruction_sft","VQA_CTRATE_diagnosis_test.json"),
            #     'w': 0.1,
            #     "task_type":"diagnosis"
            #     # "outputpath":os.path.join(self.test_output_dir,"ctrate_diagnosis.json")
            # },
            'CTRATE_Question_Answer':{
                'train':p("data","instruction_sft","VQA_CTRATE_example.json"),
                'test':p("data","instruction_sft","VQA_CTRATE_example.json"),
                'w':1.0,
                'test_file': p("data","instruction_sft","VQA_CTRATE_test.json"),
                "task_type":"text"
            },
            # 'M3D_Question_Answer':{
            #     'train':p("data","instruction_sft","VQA_M3D_open_train.json"),
            #     'test':p("data","instruction_sft","VQA_M3D_open_val.json"),
            #     'w':0.3,
            #     'test_file': p("data","instruction_sft","VQA_M3D_open_test.json"),
            #     "task_type":"text"
            # },
            # 'AbdomenCT1K_Seg':{
            #     'train':p("data","instruction_sft","VQA_SEG_AbdomenCT1K_train_new.json"),
            #     'test':p("data","instruction_sft","VQA_SEG_AbdomenCT1K_test_new.json"),
            #     'test_file':p("data","instruction_sft","VQA_SEG_AbdomenCT1K_test_new.json"),
            #     'task_type':"segmentation",
            #     'w':0.01
            # },
            # 'CTORG_Seg':{
            #     'train':p("data","instruction_sft","VQA_SEG_CTORG_train_new.json"),
            #     'test':p("data","instruction_sft","VQA_SEG_CTORG_test_new.json"),
            #     'w':0.01,
            #     "test_file":"data/instruction_sft/VQA_SEG_CTORG_test_new.json",
            #     "task_type":"segmentation"
            # },
            # 'TotalSegmentator_Organ_Seg':{
            #     'train':p("data","instruction_sft","VQA_SEG_TotalSegmentator_train_new.json"),
            #     'test':p("data","instruction_sft","VQA_SEG_TotalSegmentator_test_new.json"),
            #     'test_file':p("data","instruction_sft","VQA_SEG_TotalSegmentator_test_new.json"),
            #     'task_type':'segmentation"),
            #     'w':0.02
            # }
        }

        ### 初始化huggingface的trainingaugments
        self.training_args=TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.evaluation_strategy,  # Evaluate at the end of each epoch
            eval_steps=self.eval_steps,
            logging_first_step=self.logging_first_step,
            eval_on_start=self.eval_on_start,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            max_steps=self.num_train_iters,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_only_model=self.save_only_model,
            fp16=self.fp16,
            bf16=self.bf16,
            report_to=self.report_to,
            dataloader_num_workers=self.dataloader_num_workers,
            deepspeed=self.deepspeed_config,
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type, 
            gradient_checkpointing = self.gradient_checkpointing
        )

        self.crop_size = [256,256,32]
 
        self.vision_dim = 2048
        self.biolord_dim = 768

        if self.llm_backbone == "Qwen/Qwen2-VL-7B-Instruct":
            self.query_dim = 3584
        else:
            self.query_dim = 1536


    def get_eval_path(self, eval_iter: int):
        # 评估与测试输出目录基于 base_dir
        def p(*parts):
            return str(self.base_dir.joinpath(*parts))
        self.eval_path = p("checkpoint", self.name, f"checkpoint-{eval_iter}")
        self.test_output_dir = p("test_out", self.name, f"checkpoint-{eval_iter}")

        
if __name__=="__main__":
    training_args=qwen2vl_sftconfig()
    print(training_args.deepspeed_config)
    print(training_args.bf16)