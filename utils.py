from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
import time
import os

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):#计算出所有类名的embedding
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()  #把句子补充成句子
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)#对文本进行编码
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []#清空缓存

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for (images, target) in tqdm(train_loader_cache):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)#用CLIP的视觉编码器对图片进行编码256*1024
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)#因为每次训练的顺序是一样的，所以这里的target也是一样的，所以只需要在第一次训练的时候添加就可以了
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))#把每个augment_epoch的tensor都放到一个list里面，最后把这个list转化为tensor，然后把这个tensor的shape变成[10, 1600, 1024]，其中10是augment_epoch的数量，1600是训练集的数量，1024是视觉编码器的输出维度
        #经过数据增强之后，得到了cache_keys和cache_values，cache_keys是一个list，长度为10，里面有augment_epoch个tensor，每个tensor的shape是[1, 1600, 1024]，cache_values是一个list，里面有augment_epoch个tensor，每个tensor的shape是[256]
        # print("cache_keys的shape：",cache_keys.size)   
        
        
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)#考虑一下此处可不可以用聚类的方法来做1600*1024，考虑用聚类转化为100*1024
        
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        # temp_tensor=torch.zeros(100,1024)
        # temp_cache_values=torch.zeros(100)
        # for i in range(100):
        #     temp_tensor[i]=cache_keys[i:i+1].mean(dim=0)  #100*1024
        #     temp_cache_values[i]=torch.cat(cache_values,dim=0)[i*1]
        # cache_keys=temp_tensor.cuda().half()
        # cache_values=temp_cache_values.cuda()

        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values,dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():#不需要计算梯度
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)#64*1024
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):#寻找最佳的超参数
    year,month,day,hour,minute,second= time.localtime(time.time())[:6]
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        write_str=""
        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    adapter.eval()
                    with torch.no_grad():
                        affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # cache_logits=affinity @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
                    write_str+=str(alpha)+" "+str(beta)+" "+str(acc)+"\n"

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        save_name="VIT-B-32"+ "_"+str(cfg['shots'])+"shots_"+str(year)+"_"+str(month)+"_"+str(day)+"_"+str(hour)+"_"+str(minute)+"_"+str(second)+".txt"
        with open(os.path.join("experiment/search_a_b_acc",save_name), 'w') as f:
            f.write(write_str)
    return best_beta, best_alpha
