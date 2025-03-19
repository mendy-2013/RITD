import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.autograd import Function
from decoders.evc_blocks import LVCBlock
from decoders.evc_blocks import EVCBlock
from decoders.diffSLIC import DiffSLIC
from .spin import GenSP,SPInterAttModule,FFN
from einops import rearrange
#####重点写的模型输入口
# class gdsc(nn.Module):
#
#     def __init__(self, aux_data, in_channels, inter_channels, out_features_num=4):
#         super(gdsc, self).__init__()
#         # self.in_channels=in_channels
#         # self.inter_channels = inter_channels
#         # self.out_features_num = out_features_num
#         # self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
#         Fq ,Fk =
#
#
#
#     def forward(self, concat_x, features_list):
#
#         score = Project(concat_x)
#
#         return score
# # 输入特征图大小 HxWxC

####模型架构
class gdsc(nn.Module):
    def __init__(self):
        super(gdsc, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size= 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size= 1, bias=False)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False)
        )
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=False)
        )
        self.conv1x1 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.gen_super_pixel = GenSP(3)
        self.Linear = nn.Linear(81*128,128)
        # self.center = LVCBlock(128,128)
        # self.evc = EVCBlock(128)

    def forward_stoken(self, x, affinity_matrix):
        x = rearrange(x, 'b c h w -> b (h w) c')
        # print(x.shape,affinity_matrix.shape)
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
        return stokens

    def forward(self, x, aux_data):
        global_x = self.conv(x)
        # a = torch.any(torch.isnan(global_x))

        Fq = self.conv1(global_x)
        #####
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # conv_layer = nn.Conv2d(9, 128, kernel_size=1).to(device)
        # Fq = conv_layer(Fq)
        # b = torch.any(torch.isnan(Fq))

        Fk = self.conv2(global_x)
        # c = torch.any(torch.isnan(Fk))

        # Fc = self.conv3(global_x)

        # mask调整
        aux_data = aux_data
        # a = torch.sum(aux_data == 1).item()
        aux_data = F.interpolate(aux_data,size=(156,156), mode='bilinear', align_corners=False)
        # num_ones = np.sum(aux_data)
        # one_to_zero = int(0.5 * num_ones)
        # indices_of_ones = np.argwhere(aux_data == 1)

        # b = torch.sum(aux_data == 1).item()
        # aux_data = aux_data.repeat(1, 128, 1, 1)
        # c = torch.sum(aux_data == 1).item()
        # d = torch.sum(aux_data == 0).item()
        # print(a,b,c)

        ## 超像素特征的提取
        # slic = DiffSLIC(n_spixels=100, n_iter=5, tau=0.01, candidate_radius=1, stable=True)
        # Fk_slic= Fk
        # clst_feats, p2s_assign, s2p_assign = slic(Fk_slic)
        self.stoken_size = [16, 16]
        affinity_matrix, num_spixels = self.gen_super_pixel(global_x, self.stoken_size)

        # print(affinity_matrix.shape, num_spixels)
        stoken = self.forward_stoken(global_x, affinity_matrix)
        # print(stoken.shape)
        stoken = rearrange(stoken, 'b c k -> b (c k)')
        # print(stoken.shape)
        super_pixel = self.Linear(stoken)
        # print(super_pixel.shape,super_pixel)
        # p2s_assign = conv_layer(p2s_assign)


        # 逐元素相乘
        # print(Fq.shape)
        # print(aux_data.shape)
        ######
        elementwise_product = torch.mul(Fq, aux_data)
        # elementwise_product = torch.mul(p2s_assign, aux_data)
        # e = torch.sum(aux_data == 0).item()

        # a = torch.any(torch.isnan(elementwise_product))
        # a0 = torch.any(torch.eq(elementwise_product, 0))

        sum_result = elementwise_product.sum(dim=(2,3))
        # f = torch.sum(sum_result== 0).item()

        # mean_value = torch.mean(sum_result, dim=0)
        # std_value = torch.std(sum_result, dim=0)
        # sum_result = (sum_result - mean_value) / std_value
        # b = torch.any(torch.isnan(sum_result))
        # b0 = torch.any(torch.eq(sum_result, 0))

        sum_tensor2 = aux_data.sum(dim=(2, 3))
        # g = torch.sum(sum_result == 0).item()
        # sum_tensor2[sum_tensor2 == 0] = 0.001
        # has_zeros = torch.any(sum_tensor2 == 0).item()
        # print(has_zeros)

        # mean_value = torch.mean(sum_tensor2, dim=0)
        # std_value = torch.std(sum_tensor2, dim=0)
        # sum_tensor2 = (sum_tensor2 - mean_value) / std_value
        # c = torch.any(torch.isnan(sum_tensor2))
        # c0 = torch.any(torch.eq(sum_tensor2, 0))
        # c1 = torch.sum(aux_data == 1).item()
        # 将 sum_result 除以 sum_tensor2
        fq = sum_result / (sum_tensor2+0.0001)
        # print(fq.shape, fq)
        ###

        # print(fq.shape,fq)
        ##########
        #####中心注意力
        # fc = self.center(Fc)
        # fc = self.conv1x1(fc)
        # fc = torch.sigmoid(fc)
        # fc = F.interpolate(fc, size=(640, 640), mode='bilinear', align_corners=False)
        # ########


        # 定义线性变换层
        # linear_layer = nn.Linear(128, 512).to('cuda')
        #
        # # 进行线性变换
        # fq1 = linear_layer(fq).to('cuda')
        #######

        # fq = torch.nan_to_num(fq, nan=0)
        # mean_value = torch.mean(fq, dim=0)
        # std_value = torch.std(fq, dim=0)
        # fq = (fq - mean_value) / std_value
        # d = torch.any(torch.isnan(fq))
        # print(a)
        # print(a0)
        # print(b)
        # print(b0)
        # # print(c)
        # print(c0)
        # print(d)
        fq = fq[:, :, None, None]

        super_pixel = super_pixel[:, :, None, None]
        #####
        # Fk_sp = p2s_assign
        # Fk_sp = conv_layer(Fk_sp)
        #####
        inner_product = torch.sum(fq * Fk, dim=1)

        inner_product1 = torch.sum(super_pixel * Fk, dim=1)
        inner_product = inner_product1 + inner_product

        result = torch.sigmoid(inner_product)
        result = result.unsqueeze(1)

        #######
        # fc = fc[:, :, None, None]
        # inner_product1 = torch.sum(fc * Fk, dim=1)
        # result1 = torch.sigmoid(inner_product1)
        # result1 = result1.unsqueeze(1)
        #######
        # 两种concat
        # result = torch.cat((result, result1), dim=1)
        # result = self.conv1x1(result)
        # result = torch.sigmoid(result)
        #######
        result = F.interpolate(result, size=(640, 640), mode='bilinear', align_corners=False)


        # 反卷积
        # conv_transpose = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=3, padding=1, output_padding=2)
        # conv_transpose = conv_transpose.to('cuda')
        # # 将 tensor 调整为合适的形状
        # result = result.unsqueeze(1)
        # # 进行反卷积操作
        # result = conv_transpose(result, output_size=(640, 640))


        # 将结果中的每个通道相加

        # has_any_nan3 = torch.any(torch.isnan(result))

        return result
        # return result, fc
        # return result, fq1

# if __name__ == "__main__":
#     aux_data = torch.ones((3,640,640))
#     x = torch.randn((1,256,64,64))
#     model = gdsc()
#     y = model(x,aux_data)
#     print(y.shape)






