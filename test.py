from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import time
from dataset import MyDataSet
from STN import AffineTransform
from loss import ComputeLoss


def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig('./%s' % name)
    plt.show()


def test():
    print('Using device ' + str(device) + ' for training!')
    scale1_model.eval()
    scale2_model.eval()
    scale3_model.eval()
    for i, data in enumerate(dataloader):
        ref_tensor = data[0]
        sen_tensor = data[1]
        "Scale: 1"
        scale_1_affine_parameter = scale1_model(ref_tensor, sen_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_1 = AffineTransform(ref_tensor, sen_tensor,
                                                                                  scale_1_affine_parameter)
        loss_1 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)

        "Scale: 2"
        scale_2_affine_parameter = scale2_model(ref_tensor, sen_tran_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_2 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                  scale_2_affine_parameter)
        loss_2 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)

        "Scale: 3"
        scale_3_affine_parameter = scale3_model(ref_tensor, sen_tran_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_3 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                  scale_3_affine_parameter)
        # inv_affine_parameter = torch.matmul(torch.matmul(inv_affine_parameter_1, inv_affine_parameter_2),
        #                                     inv_affine_parameter_3)
        loss_3 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        loss = 0.14285714 * loss_1 + 0.28571429 * loss_2 + 0.57142857 * loss_3

        if i % 50 == 0:
            print('%f%% loss: %f' % ( i / total_epoch * 100, loss))



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''''''
    data_path = ''
    batch_size = 1
    scale1_model = torch.load('')
    scale2_model = torch.load('')
    scale3_model = torch.load('')
    ''''''
    dataset = MyDataSet('test', data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    total_epoch = len(dataloader)
    model_save_path = os.path.join(data_path, 'train', 'save_model')
    test()
