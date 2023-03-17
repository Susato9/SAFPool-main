import os
os.sys.path.append('main_miniImageNet.py')

import random
import argparse
import yaml
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from clip.model import linearCLIP
# from datasets.imagenet import ImageNet
from datasets.mini_ImageNet import MiniImageNet
import clip
from utils import *


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format', type=str, default='configs/miniImageNet.yaml')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    #test_features:12000*1024  test_labels:12000   其中12000为测试集的样本数，1024为clip的特征维度
    #cache_keys:1024*1600       cache_values:1600*100   其中1600为缓存的样本数，100为类别数，1024为clip的特征维度
    #clip_weights:1024*100
    
    # Zero-shot CLIP  直接用clip的文本特征与测试集的特征进行比较
    clip_logits = 100. * test_features @ clip_weights  #就是比较测试集的特征与clip文本特征的相似度
    # print(clip_logits.shape)
    # print(test_labels.shape)
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys#比较测试集特征与缓存特征的相似度
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    # print(clip_logits.shape, cache_logits.shape)
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)#在给定的beta和alpha的范围内，找到最优的beta和alpha


def run_tip_adapter_F(cfg, cache_keys, cache_values, test_loader, test_labels, datasets, model, train_loader_F):
    
    # Enable the cached keys to be learnable
    # adapter1 = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.dtype).cuda()
    # adapter1.weight = nn.Parameter(cache_keys.t())
    
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    clip_weights = clip_classifier(datasets.classnames1, datasets.template, model)
    txt_name=str(cfg["backbone"])+" "+str(cfg["shots"])+" "+str(now_time)+'.txt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    
    # for name, param in model.named_parameters():
    #     print(name)
    #         #  if 'adapter' not in name:
    #         #      param.requires_grad_(False)
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    print(model)
    best_acc, best_epoch = 0.0, 0
    

    for train_idx in range(cfg['train_epoch']):
        # Train
        # adapter1.train()#进入训练模式
        model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            # with torch.no_grad():
            image_features = model.encode_image(images)#256*1024
            image_features =image_features/ image_features.norm(dim=-1, keepdim=True)

            # affinity = adapter1(image_features)#可以把两个矩阵相乘理解为矩阵通过一个线性变换后的结果
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            cache_logits=0
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        # adapter1.eval()
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images, target = images.cuda(), target.cuda()
                # with torch.no_grad():
                image_features = model.encode_image(images)#256*1024
                image_features =image_features/ image_features.norm(dim=-1, keepdim=True)

                # affinity = adapter1(image_features)#可以把两个矩阵相乘理解为矩阵通过一个线性变换后的结果
                # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                cache_logits=0
                clip_logits = 100. * image_features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha

                # loss = F.cross_entropy(tip_logits, target)

                acc += cls_acc(tip_logits, target)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = train_idx
        # for file in 
        # # affinity = adapter1(test_features)
        # # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # cache_logit=0
        # clip_logits = 100. * test_features @ clip_weights
        # tip_logits = clip_logits + cache_logits * alpha
        # acc = cls_acc(tip_logits, test_labels)
        
        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc/(len(test_loader))))
        save_root=os.path.join("experiment/add_linear_images",txt_name)
        with open(save_root,"a") as f:
            f.write("{}\n".format(acc/(len(test_loader))))
        # if acc > best_acc:
        #     best_acc = acc
        #     best_epoch = train_idx
        #     # torch.save(model.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    # model.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter1)


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)#导入yaml文件

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir#创建缓存模型储存文件

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()#载入CLIP模型
    model=linearCLIP(clip_model).cuda()
    # 载入MiniImageNet数据集
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    datasets = MiniImageNet(cfg['root_path'], cfg['shots'], preprocess)


    test_loader = torch.utils.data.DataLoader(datasets.test1, batch_size=64, num_workers=8, shuffle=False)
    train_loader_cache = torch.utils.data.DataLoader(datasets.train, batch_size=256, num_workers=8, shuffle=False,drop_last=False)
    train_loader_F = torch.utils.data.DataLoader(datasets.train, batch_size=64, num_workers=8, shuffle=True, drop_last=False)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    # clip_weights = clip_classifier(datasets.classnames1, datasets.template, model)#获取文本特征1024*100
    # print("Textual features shape ", clip_weights.shape)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, model, train_loader_cache)#构建缓存器模型
    ##cache_keys:1024*1600  cache_values:1600*100
    
    # print("Cache model's keys shape ", cache_keys.shape)
    # print("Cache model's values shape ", cache_values.shape)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", model, test_loader)#载入测试数据被CLIP模型处理后的特征
    #test_features:12000*1024  test_labels:12000
    # print(test_features.shape, test_labels.shape)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    # run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_loader, test_labels, datasets, model, train_loader_F)
           

if __name__ == '__main__':
    main()