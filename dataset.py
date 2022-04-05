import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataSet(Dataset):
    def __init__(self, mode, data_path):
        self.mode = mode
        if self.mode == 'train':
            # self.data_path = os.path.join(data_path, 'train')
            self.data_path = data_path
            self.imginfo_path = os.path.join(self.data_path, 'image_pair.txt')
            self.img_and_gt = read_label(self.imginfo_path)
            self.ref_path = os.path.join(self.data_path, 'reference')
            self.sen_path = os.path.join(self.data_path, 'sensed')

        if self.mode == 'test':
            self.data_path = os.path.join(data_path, 'test')
            self.imgpair_info = os.path.join(self.data_path, 'image_pair.txt')
            self.img_and_gt = read_label(self.imginfo_path)
            self.ref_path = os.path.join(self.data_path, 'reference')
            self.sen_path = os.path.join(self.data_path, 'sensed')

    def __len__(self):
        return len(self.img_and_gt)

    def __getitem__(self, index):
        self.ref_name = self.img_and_gt[index][0]
        self.sen_name = self.img_and_gt[index][1]
        self.gt_tps = list(map(float, self.img_and_gt[index][2:8]))
        self.ref_pil = Image.open(os.path.join(self.ref_path, self.ref_name))
        self.sen_pil = Image.open(os.path.join(self.sen_path, self.sen_name))
        self.ref_tensor = pil_to_tensor(self.ref_pil)
        self.sen_tensor = pil_to_tensor(self.sen_pil)
        self.gt_tps_tensor = torch.Tensor(self.gt_tps).to(device)
        return self.ref_tensor, self.sen_tensor, self.gt_tps_tensor


def read_label(label_path):
    label = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str in data:
                sub_str = str.split(' ')
            if sub_str:
                label.append(sub_str)
    return label


def pil_to_tensor(p):
    if p.mode != 'L':
        p = p.convert('L')
    return torch.tensor(np.float32(np.array(p))).to(device).unsqueeze(0)