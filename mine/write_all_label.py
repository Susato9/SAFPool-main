import numpy as np

with open('mine/name_label/all_name_list.txt', 'r') as f:
    name_list=""
    lines = f.readlines()
    for line in lines:
        if line!="\n":
            name = line.split(' ')[1]
            label = line.split(' ')[-1].strip()
            name_list += name + " " + label + "\n"


with open('mine/name_label/all_label_list.txt', 'w') as f:
    f.write(name_list)
            