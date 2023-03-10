import numpy as np
import os


def create_dict():
    classname_dict={}
    with open("mine/name_label/all_label_list.txt",'r') as f:
        lines=f.readlines()
        for line in lines:
            num=line.split(' ')[0]
            name=line.strip().split(' ')[1]
            classname_dict[num]=name
    return classname_dict
    
def find_classname(txt,path,dict_temp):
    for file in sorted(os.listdir(path)):
        if file in sorted(dict_temp.keys()):
            with open(txt,'a') as f:
                f.write(file+" "+dict_temp[file]+"\n")


if __name__ == "__main__":
    dict_temp=create_dict()
    print(dict_temp)
    txt="mine/name_label/test_classname.txt"
    path="DATA/imagenet/miniImageNet_orginize/test"
    find_classname(txt,path,dict_temp)
    