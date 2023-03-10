import os

from datasets.utils import  DatasetBase

from torchvision.datasets import ImageFolder as imageFolder
from collections import defaultdict
import random

template = ['a photo of {}, a type of food.']
from torchvision import transforms

imagenet_classes = ["roundworm",
"camtschatica",
"retriever",
"malamute",
"dog",
"pictus",
"leo",
"pismire",
"nigripes",
"bookstall",
"crate",
"cuirass",
"guitar",
"hourglass",
"bowl",
"bus",
"scoreboard",
"curtain",
"vase",
"trifle"
]

imagenet_templates = ["itap of a {}.",
                        "a bad photo of the {}.",
                        "a origami {}.",
                        "a photo of the large {}.",
                        "a {} in a video game.",
                        "art of the {}.",
                        "a photo of the small {}."]

class MiniImageNet(DatasetBase):

       def __init__(self, root, num_shots, preprocess):
        self.dataset_dir = root
        train_preprocess = transforms.Compose([transforms.Resize(256),#缩放到256
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),#随机中心裁剪
                                                transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转
                                                transforms.ToTensor(),#转换为tensor
                                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])#训练数据增广方式
        test_preprocess = preprocess

        train_root=os.path.join(self.dataset_dir,'test')
        val_root=os.path.join(self.dataset_dir,'test')
        # test_root=os.path.join(self.dataset_dir,'test')
        # train_root=self.dataset_dir
        
        self.train=imageFolder(train_root,transform=train_preprocess)
        self.val1=imageFolder(val_root,transform=test_preprocess)
        # self.test1=imageFolder(val_root,transform=test_preprocess)
        
        self.template = imagenet_templates  #prompt 的模板
        self.classnames1 = imagenet_classes #类别名称

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []
        
        q_imgs=[]
        q_targets=[]

        # print(split_by_label_dict.items(),type(split_by_label_dict.items()))
        
        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, num_shots)#随机从训练集中选取num_shots个样本
            targets = targets + [label for i in range(num_shots)]#抽出其中对应的标签
            # print(items,type(items))
            
            item_temp=list(set(items)-set(imgs))
            # q_imgs = q_imgs + random.sample(item_temp, num_shots)#随机从训练集中选取num_shot个样本
            q_targets = q_targets + [label for i in range(len(items)-num_shots)]#抽出其中对应的标签
        

        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs
        
        
        self.support_set = self.train
        self.query_set = self.val1
        
        self.query_set.imgs=q_imgs
        self.query_set.targets=q_targets
        self.query_set.samples=q_imgs
    
if __name__ == '__main__':
    preprocess=transforms.Compose([
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
    DATA=MiniImageNet(root='DATA/imagenet/miniImageNet_orginize', num_shots=1, preprocess=preprocess)
    
    print(DATA.support_set.imgs[1])
    print(DATA.query_set.imgs[1])
    