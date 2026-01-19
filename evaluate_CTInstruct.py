import torch
import os 
import json 
import numpy as np
from torch.utils.data import DataLoader 
from safetensors.torch import load_file
from time import time

from transformers import AutoTokenizer
from dataset.qwvl2_dataloader_clip_seg import *
from dataset.qwen_processor_clip import Qwen2VLProcessor
from model.llm_vision_feature_generator import LLM_Vision_Encoder
from model.Qwen import Qwen2VLForConditionalGeneration

from config.sft_config_SFT_multi_task_final_eval import qwen2vl_sftconfig
from metric.get_metric import *
import random
import evaluate


def load_safetensor_weights(model, checkpoint_dir):
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    state_dict = {}
    with open(index_file, "r") as f:
        index = json.load(f)
    for key, value in index["weight_map"].items():
        slice_state_dict = load_file(os.path.join(checkpoint_dir, value))   
        state_dict[key] = slice_state_dict[key]
    model.load_state_dict(state_dict)
    return model

def prepare_dataset(config, dataloader_params):
    test_iters_list=[]
    for task_name in config.task_list:
        task = config.task_list[task_name]
        test_dataset=Qwenvl2_SFTDataset(task['test_file'])
        eval_dataloader = DataLoader(test_dataset, **dataloader_params)
        test_iters_list.append([iter(eval_dataloader),len(test_dataset),task["task_type"],task_name])
    return test_iters_list
    
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def to_cuda(variable):
    if not isinstance(variable,list):
        if type(variable) is np.ndarray:
            variable = torch.tensor(variable)
        return variable.cuda()
    for idx in range(len(variable)):
        variable[idx] = to_cuda(variable[idx])
    return variable

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

config=qwen2vl_sftconfig()

#####################  Processor & Tokenizer Loading  ####################
print("Initialze Processor and Tokenizer")
processor = Qwen2VLProcessor.from_pretrained(
    config.llm_backbone,
    cache_dir=config.cache_dir)
processor.crop_size = config.crop_size

tokenizer = AutoTokenizer.from_pretrained(config.eval_path, trust_remote_code=False)

processor.tokenizer = tokenizer
config.seg_token_idx = processor.tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
config.disease_conf_token_idx = processor.tokenizer("<disease_confidence>", add_special_tokens=False).input_ids[0]

###########################  Dataset Loading  ############################
print("Initialize Dataset")

data_collator=TestQwenvl2_ModelCollator(processor=processor, crop_size=config.crop_size, patch_max=config.patch_max)
dataloader_params = {
    "batch_size": 1,
    "collate_fn": data_collator,
    "num_workers": 0,  # 16
    'shuffle': False,
    'drop_last': False
}
test_iters_list = prepare_dataset(config, dataloader_params)

###########################  Model Loading  ############################
print("Initialize Model")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.eval_path,
    cache_dir=config.cache_dir,
    torch_dtype=torch.float32,
    attn_implementation="flash_attention_2")

model.visual = LLM_Vision_Encoder(config).cuda()

model = load_safetensor_weights(model, config.eval_path)

model.resize_token_embeddings(len(tokenizer))
model.seg_token_idx = config.seg_token_idx
model.disease_conf_token_idx = config.disease_conf_token_idx
model.freeze_vision = False

print("tokenizer length",len(tokenizer))
print("pad_token",tokenizer.pad_token)
print("padding_side",tokenizer.padding_side)

random_seed(seed=42)

model = model.cuda()
model.eval()
os.makedirs(config.test_output_dir,exist_ok=True)

generate_length = 4096

for eval_iters, dataset_length, task_type, task_name in test_iters_list:
    outputpath = os.path.join(config.test_output_dir, task_name+".json")
    if task_type == "text":
        meteor_scores = []
        bert_scores = []
        bleu_scores = []
        rouge_scores = []
        savedata = {}
    elif task_type == "diagnosis":
        if 'radchest' in outputpath.lower():
            disease_list = json.load(open('./data/radchest_label.json','r'))
        else:
            disease_list = json.load(open('./data/ctrate_label.json','r'))
        
        all_pred_prob = []
        all_y_true = []
        
        savedata = {}

    elif task_type == "segmentation":
        dices = []
        savedata = {}
        seg_root_dir = os.path.join(config.test_output_dir,"mask")
        os.makedirs(seg_root_dir,exist_ok=True)

    for idx in range(dataset_length):
        sample={}
        print(idx+1,"/",dataset_length)
        inputs = next(eval_iters)
        cur_inputs = {}
        
        for tk in inputs.keys():
            if tk == "answer" or tk == "image_path_list" or tk == "question" or "choose_label" in tk:
                continue
            if tk != "patch_num":
                cur_inputs[tk] = to_cuda(inputs[tk])
            else:
                cur_inputs[tk] = inputs[tk]

        if task_type == "text":

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generate_ids = model.generate(**cur_inputs, 
                    max_new_tokens=generate_length,
                    num_beams=5,
                    top_k=50,
                    temperature=0.7,
                )
            
            preds=processor.batch_decode(generate_ids, skip_special_tokens=True, max_new_tokens=generate_length)
            gen_text = []
            for pred in preds:
                parts = pred.split("assistant\n", 1)
                if len(parts) > 1:
                    result = parts[1]
                else:
                    result = ""
                gen_text.append(result)
            
            sample["predict"]=gen_text
            sample["ground_truth"]=[gt.strip() for gt in inputs['answer']]
            sample["image_path_list"] = inputs["image_path_list"]
            sample["question"] = inputs['question']

            decoded_preds = sample["predict"]
            decoded_labels = [[label] for label in sample["ground_truth"]]
            bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            bleu_scores.append(bleu_score['bleu'])

            rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
            rouge_scores.append(rouge_score['rouge1'])

            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            meteor_scores.append(meteor_score['meteor'])

            avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
            avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

            std_meteor = np.std(meteor_scores) if meteor_scores else 0
            std_bert = np.std(bert_scores) if bert_scores else 0
            std_bleu = np.std(bleu_scores) if bleu_scores else 0
            std_rouge = np.std(rouge_scores) if rouge_scores else 0
            metric = {}
            metric['meteor'] = {'avg':avg_meteor, 'std':std_meteor}
            metric['bleu'] = {'avg':avg_bleu, 'std':std_bleu}
            metric['rouge'] = {'avg':avg_rouge, 'std':std_rouge}
            metric['bert'] = {'avg':avg_bleu, 'std':std_bleu}

            savedata['metric'] = metric


        elif task_type == "diagnosis":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(**cur_inputs,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=generate_length
                )
            generate_ids = outputs.sequences
            hidden_states = outputs.hidden_states
            dic = {}
            last_layer_hidden_states = [state[-1] for state in hidden_states]
            final_hidden_state = torch.cat(last_layer_hidden_states, dim=1)
            
            targets = cur_inputs['targets']

            disease_token_mask = generate_ids == config.disease_conf_token_idx

            if disease_token_mask.shape[1] > final_hidden_state.shape[1]:
                disease_token_mask = disease_token_mask[:, :-1]
            if final_hidden_state.shape[1] > disease_token_mask.shape[1]:
                final_hidden_state = final_hidden_state[:, :-1]

            n_dis_tokens = disease_token_mask.sum().item() # the seg num in the whole batch
            
            if n_dis_tokens > 0:
                diag_latent_embeddings = final_hidden_state[disease_token_mask]
                if len(disease_list) == 16:
                    logits = torch.concat([head(diag_latent_embeddings[head_index]) for head_index,head in enumerate(model.rad_class_heads)])
                else:
                    logits = torch.concat([head(diag_latent_embeddings[head_index]) for head_index,head in enumerate(model.ctrate_class_heads)])
                all_pred_prob.append(torch.sigmoid(logits).detach().cpu().float().numpy())
                all_y_true.append(targets.detach().cpu().float().numpy())

        elif task_type == "segmentation":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(**cur_inputs,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=generate_length
                )
            generate_ids = outputs.sequences
            hidden_states = outputs.hidden_states
            dic = {}
            last_layer_hidden_states = [state[-1] for state in hidden_states]
            final_hidden_state = torch.cat(last_layer_hidden_states, dim=1)

            gt_masks = cur_inputs['masks']

            seg_token_mask = generate_ids == config.seg_token_idx # B, seq_length
            if seg_token_mask.shape[1] > final_hidden_state.shape[1]:
                seg_token_mask = seg_token_mask[:, :-1]
            if final_hidden_state.shape[1] > seg_token_mask.shape[1]:
                final_hidden_state = final_hidden_state[:, :-1]

            n_seg_tokens = seg_token_mask.sum().item() # the seg num in the whole batch

            s1 = time()
            
            if n_seg_tokens > 0:
                # generate segmentations and calculate dice loss

                if len(final_hidden_state) > 1:
                    seg_latent_embeddings_list = []
                    for beam_hidden_states in final_hidden_state:
                        seg_latent_embeddings_list.append(beam_hidden_states[None,:][seg_token_mask])
                else:
                    seg_latent_embeddings_list = [final_hidden_state[seg_token_mask]]

                max_dice = 0
                for seg_latent_embeddings in seg_latent_embeddings_list:

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        pred_masks = model.visual.decode_mask(cur_inputs['pixel_values'], cur_inputs['marks'], cur_inputs['patch_num'],seg_latent_embeddings, model.device,True)
                    
                    dice = cal_dice(pred_masks, gt_masks)
                    max_dice = max(dice,max_dice)
                
                dices.append(max_dice)
                sample['dice'] = str(max_dice)
            

            gen_text=processor.batch_decode(generate_ids, skip_special_tokens=True, max_new_tokens=generate_length)
            sample["ground_truth"]=inputs['answer']
            sample["predict"]=gen_text
            sample['image_path'] = inputs['image_path_list'][0][0]
            sample['question'] = inputs['question']
            metric = {"dice":{'mean':str(np.mean(dices)),'std':str(np.std(dices))}}
            savedata['metric'] = metric
        
        gen_text=processor.batch_decode(generate_ids, skip_special_tokens=True, max_new_tokens=generate_length)
        sample["ground_truth"]=inputs['answer']
        sample["predict"]=gen_text
        sample['image_path'] = inputs['image_path_list'][0][0]
        sample['question'] = inputs['question']
        savedata[str(idx+1)] = sample
        with open(outputpath, 'w', encoding='utf-8') as f:
            json.dump(savedata, f,indent=4)
    
    if task_type == "diagnosis":
        metric = {'precision':0,'f1':0,'acc':0,'auc':0,'thresh':0,'per_disease':{d:{} for d in disease_list}}
        aucs,accs,precisions,f1s,threshs = get_diagnosis_metric_bce(all_y_true, all_pred_prob)
        for d_idx,d in enumerate(disease_list):
            metric['per_disease'][d]['precision'] = precisions[d_idx]
            metric['per_disease'][d]['f1'] = f1s[d_idx]
            metric['per_disease'][d]['acc'] = accs[d_idx]
            metric['per_disease'][d]['auc'] = aucs[d_idx]
            metric['per_disease'][d]['thresh'] = threshs[d_idx]
        metric['precision'] = {'mean':np.mean([metric['per_disease'][d]['precision'] for d in metric['per_disease']]),'std':np.std([metric['per_disease'][d]['precision'] for d in metric['per_disease']])}
        metric['f1'] = {'mean':np.mean([metric['per_disease'][d]['f1'] for d in metric['per_disease']]),'std':np.std([metric['per_disease'][d]['f1'] for d in metric['per_disease']])}
        metric['acc'] = {'mean':np.mean([metric['per_disease'][d]['acc'] for d in metric['per_disease']]),'std':np.std([metric['per_disease'][d]['acc'] for d in metric['per_disease']])}
        metric['auc'] = {'mean':np.mean([metric['per_disease'][d]['auc'] for d in metric['per_disease']]),'std':np.std([metric['per_disease'][d]['auc'] for d in metric['per_disease']])}
        metric['thresh'] = {'mean':np.mean([metric['per_disease'][d]['thresh'] for d in metric['per_disease']]),'std':np.std([metric['per_disease'][d]['thresh'] for d in metric['per_disease']])}
    savedata['metric'] = metric
    with open(outputpath, 'w', encoding='utf-8') as f:
        json.dump(savedata, f,indent=4)