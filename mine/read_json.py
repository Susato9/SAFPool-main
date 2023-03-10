import json

with open('classes_name.json','r', encoding='gbk') as f: 
	json_file = json.loads(f.read())

name_str=""
for num in sorted(json_file.keys()):
    name_str+="\""+json_file[num][1]+"\",\n"

with open("mine/name_label/new_name_str.txt", "w") as f:
    f.write(name_str)
    
    