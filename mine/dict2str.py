import numpy as np

def dict2str(source_txt, target_txt):
    with open(source_txt, 'r') as f:
        name_list=""
        lines = f.readlines()
        for line in lines:
            if line!="\n":
                name = line.split(' ')[1]
                label = line.split(' ')[-1].strip()
                name_list +='\"'+label +"\","+ "\n"
    with open(target_txt, 'w') as f:
        f.write(name_list)
        

if __name__ == "__main__":
    source_txt="mine/name_label/test_classname.txt"
    target_txt="mine/name_label/test_name.txt"
    dict2str(source_txt, target_txt)