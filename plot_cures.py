import numpy as np
import matplotlib.pyplot as plt

np1_root="learing_rate.txt"
# np2_root="experiment/ori_model/RN101_16_shots_2023_3_22_13_51_52.txt"
# np3_root="experiment/ori_model/vit_B_32_16_shots_2023_3_22_15_16_56.txt"
# np4_root="experiment/ori_model/RN50_1_shots_2023_3_22_10_22_58.txt"
# np5_root="experiment/ori_model/RN50_5_shots_2023_3_22_10_29_9.txt"
# np6_root="experiment/ori_model/RN101_1_shots_2023_3_22_11_45_0.txt"
# np7_root="experiment/ori_model/RN101_5_shots_2023_3_22_11_51_39.txt"
# np8_root="experiment/ori_model/vit_B_32_1_shots_2023_3_22_22_17_0.txt"
# np9_root="experiment/ori_model/vit_B_32_5_shots_2023_3_22_14_32_42.txt"

def read_txt(root):
    with open(root, "r") as f:
        lines = f.readlines()
        lines = [float(line.strip()) for line in lines]
    return lines

np1=np.array(read_txt(np1_root))
# np2=np.array(read_txt(np2_root))
# np3=np.array(read_txt(np3_root))

# x=np.arange(1, 21, 1)
# plt.plot(x,np1, label="1-shot")
# plt.plot(x,np2, label="5-shot")
# plt.plot(x,np3, label="16-shot")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy/%")
# plt.xticks(np.arange(1,21,1))
# # plt.ylim(60,70)
# plt.title("the accuracy of different shots of vit B/32")
# plt.legend()
# plt.savefig("vit_B_32.png",dpi=int(300))


# np1=np.array(read_txt(np1_root))
# np2=np.array(read_txt(np2_root))
# np3=np.array(read_txt(np3_root))
# np4=np.array(read_txt(np4_root))
# np5=np.array(read_txt(np5_root))
# np6=np.array(read_txt(np6_root))
# np7=np.array(read_txt(np7_root))
# np8=np.array(read_txt(np8_root))
# np9=np.array(read_txt(np9_root))

x=np.arange(1, 21, 1)
plt.plot(x,np1, label="learning rate")
# plt.plot(x,np2, label="RN101-16-shot")
# plt.plot(x,np3, label="vit B/32-16-shot")
# plt.plot(x,np4, label="RN50-1-shot")
# plt.plot(x,np5, label="RN50-5-shot")
# plt.plot(x,np6, label="RN101-1-shot")
# plt.plot(x,np7, label="RN101-5-shot")
# plt.plot(x,np8, label="vit B/32-1-shot")
# plt.plot(x,np9, label="vit B/32-5-shot")

plt.xlabel("Epoch")
plt.ylabel("learning rate")
plt.xticks(np.arange(1,21,1))
# plt.ylim(60,70)
plt.title("CosineAnnealingLR")
plt.legend()
plt.savefig("learning_rate",dpi=int(300))