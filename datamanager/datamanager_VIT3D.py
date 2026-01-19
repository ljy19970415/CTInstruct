import json
from dataset_VIT.dataset_ct_3dseg_clip import Segmentation_trainDataset, DataCollator_seg
from dataset_VIT.dataset_ct_3dcls_clip import Classification_Dataset, DataCollator_cls
from dataset_VIT.dataset_ct_imagetext import imageText_Dataset, DataCollator_imagetext
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class data_Manager():
    def __init__(self,task_names,meta_data_dic,batchsize,num_workers=0,crop_size=(256,256,96), patch_max=6, flag_2D_dic=None):
        self.crop_size=crop_size
        self.traindata_dict={}
        self.trainloader_dict={}
        self.testdata_dict={}
        self.testloader_dict={}
        self.trainiter_dict={}
        self.testiter_dict={}
        self.num_workers=num_workers
        self.batchsize=batchsize
        self.task_names=task_names
        #self.numclass_dict=numclass_dict
        self.meta_data_dict=meta_data_dic
        self.patch_max = patch_max
        self.flag_2D_dic = flag_2D_dic
        self.get_data()
        
    def get_data(self):
        task_names=self.task_names
        #print(self.batchsize_dict)
        batchsize=self.batchsize

        image_text_data_collator = DataCollator_imagetext()
        seg_data_collator = DataCollator_seg()
        cls_data_collator = DataCollator_cls()

        for task_name in task_names:

            #batchsize=self.batchsize_dict[task_name]
            metadata=self.meta_data_dict[task_name]
            instructions = json.load(open(metadata,'r'))
            train_data=instructions["train"]
            test_data=instructions["test"]

            print("Number of training data: ",len(train_data),flush=True)
            print("Number of testing data: ",len(test_data),flush=True)

            if "segmentation" in task_name:
                
                train_dataset = Segmentation_trainDataset(task_name,train_data, self.crop_size, True, self.patch_max)
                test_dataset = Segmentation_trainDataset(task_name,test_data, self.crop_size, False, self.patch_max)
                train_loader = DataLoader(train_dataset, self.batchsize['segmentation'],sampler=DistributedSampler(train_dataset, shuffle=True),
                                    num_workers=self.num_workers, pin_memory=False,drop_last=True,collate_fn=seg_data_collator)
                test_loader = DataLoader(test_dataset, self.batchsize['segmentation'], shuffle=True, num_workers=self.num_workers, drop_last=False, collate_fn=seg_data_collator)
            
            elif "classification" in task_name or "treatment_planning" in task_name:
                
                train_dataset=Classification_Dataset(task_name,train_data,self.crop_size,True,self.patch_max,flag_2D=self.flag_2D_dic[task_name])
                test_dataset=Classification_Dataset(task_name,test_data,self.crop_size,False,self.patch_max,flag_2D=self.flag_2D_dic[task_name])
                train_loader = DataLoader(train_dataset, batchsize['classification'][task_name],sampler=DistributedSampler(train_dataset, shuffle=True),
                                    num_workers=self.num_workers, pin_memory=True,drop_last=True, collate_fn=cls_data_collator)
                test_loader = DataLoader(test_dataset, batch_size=batchsize['classification'][task_name], shuffle=True, num_workers=self.num_workers,drop_last=False,collate_fn=cls_data_collator)
            
            elif 'image_text' in task_name:
                train_dataset = imageText_Dataset(task_name,train_data,self.crop_size,aug=True, patch_max=self.patch_max)
                test_dataset = imageText_Dataset(task_name,test_data,self.crop_size,aug=False, patch_max=self.patch_max)
                train_loader = DataLoader(train_dataset, self.batchsize['image_text'],sampler=DistributedSampler(train_dataset, shuffle=True),
                                    num_workers=self.num_workers, pin_memory=True,drop_last=True,collate_fn=image_text_data_collator)
                test_loader = DataLoader(test_dataset, self.batchsize['image_text'], shuffle=True, num_workers=self.num_workers, drop_last=False,collate_fn=image_text_data_collator)
            
            self.traindata_dict[task_name]=train_dataset
            self.trainloader_dict[task_name]=train_loader
            self.testdata_dict[task_name]=test_dataset
            self.testloader_dict[task_name]=test_loader
            self.trainiter_dict[task_name]=iter(train_loader)
            self.testiter_dict[task_name]=iter(test_loader)