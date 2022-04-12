from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from dataset import pil_to_tensor
from STN import AffineTransform

def show_tensor(t, a, b, c, tt):
    t = t.squeeze(0).squeeze(0)
    t = np.array(t.detach().cpu())
    plt.subplot(a,b,c)
    plt.title(tt)
    plt.imshow(t, cmap='gray')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_1_path = ''  # reference image path: 'xx.jpg'
    image_2_path = ''  # warped sensed image path: 'xx.jpg'
    model_path = ''  # 'xx.pth'

    image_1 = pil_to_tensor(Image.open(image_1_path)).unsqueeze(0)
    image_2 = pil_to_tensor(Image.open(image_2_path)).unsqueeze(0)
    model = torch.load(model_path)
    tps = model(image_1, image_2)
    image_2_correct, image_1_warp, _ = AffineTransform(image_1, image_2, tps)
    show_tensor(image_1, 1, 3, 1, 'input_ref')
    show_tensor(image_2, 1, 3, 2, 'input_sen')
    show_tensor(image_2_correct, 1, 3, 3, 'output_correct_sen')
    plt.show()