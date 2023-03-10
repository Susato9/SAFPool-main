import os
from datasets.utils import  DatasetBase

from torchvision.datasets import ImageFolder as imageFolder
from collections import defaultdict
import random

template = ['a photo of {}, a type of food.']
from torchvision import transforms

imagenet_classes = ["house_finch",
"robin",
"triceratops",
"green_mamba",
"harvestman",
"toucan",
"goose",
"jellyfish",
"nematode",
"king_crab",
"dugong",
"Walker_hound",
"Ibizan_hound",
"Saluki",
"golden_retriever",
"Gordon_setter",
"komondor",
"boxer",
"Tibetan_mastiff",
"French_bulldog",
"malamute",
"dalmatian",
"Newfoundland",
"miniature_poodle",
"white_wolf",
"African_hunting_dog",
"Arctic_fox",
"lion",
"meerkat",
"ladybug",
"rhinoceros_beetle",
"ant",
"black-footed_ferret",
"three-toed_sloth",
"rock_beauty",
"aircraft_carrier",
"ashcan",
"barrel",
"beer_bottle",
"bookshop",
"cannon",
"carousel",
"carton",
"catamaran",
"chime",
"clog",
"cocktail_shaker",
"combination_lock",
"crate",
"cuirass",
"dishrag",
"dome",
"electric_guitar",
"file",
"fire_screen",
"frying_pan",
"garbage_truck",
"hair_slide",
"holster",
"horizontal_bar",
"hourglass",
"iPod",
"lipstick",
"miniskirt",
"missile",
"mixing_bowl",
"oboe",
"organ",
"parallel_bars",
"pencil_box",
"photocopier",
"poncho",
"prayer_rug",
"reel",
"school_bus",
"scoreboard",
"slot",
"snorkel",
"solar_dish",
"spider_web",
"stage",
"tank",
"theater_curtain",
"tile_roof",
"tobacco_shop",
"unicycle",
"upright",
"vase",
"wok",
"worm_fence",
"yawl",
"street_sign",
"consomme",
"trifle",
"hotdog",
"orange",
"cliff",
"coral_reef",
"bolete",
"ear"
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

        train_root=os.path.join(self.dataset_dir,'train')
        val_root=os.path.join(self.dataset_dir,'val')
        # test_root=os.path.join(self.dataset_dir,'test')
        # train_root=self.dataset_dir
        
        self.train=imageFolder(train_root,transform=train_preprocess)
        self.val1=imageFolder(val_root,transform=test_preprocess)
        self.test1=imageFolder(val_root,transform=test_preprocess)
        
        self.template = imagenet_templates  #prompt 的模板
        self.classnames1 = imagenet_classes #类别名称

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []

        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, num_shots)#随机从训练集中选取num_shots个样本
            targets = targets + [label for i in range(num_shots)]#抽出其中对应的标签
            print(imgs)
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs
    
if __name__ == '__main__':
    preprocess=transforms.Compose([
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
    DATA=MiniImageNet(root='DATA/imagenet/miniImageNet_orginize', num_shots=1, preprocess=preprocess)
    