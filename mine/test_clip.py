import os,sys
sys.path.append(os.getcwd())
import numpy as np

import clip
from PIL import Image
import torch


model,preprocess=clip.load("RN50")
image=Image.open("mine/dog.bmp").convert("RGB").resize((224,224))
image=preprocess(image).cuda()
print(image.shape)
image=image.unsqueeze(0)
print(image.shape)
# image=torch.tensor((np.array(image))).unsqueeze(0).permute(0,3,1,2).float().cuda()
# image=preprocess(image).cuda()



classname=["dog","cat","people","book","Golden retriever","retriever"]
text_description = [f"a photo of a {text}." for text in classname]
text_tokens = clip.tokenize(text_description).cuda()

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)#归一化(4*1024)
    
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)#归一化1*1024
    

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
label_idx = similarity.argmax(dim=-1)
print("该图片是一张",classname[label_idx],"的图片")



