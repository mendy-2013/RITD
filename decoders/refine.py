import torch
import torch.nn as nn
import torch.nn.functional as F
from decoders.cross_attention import SPInterAttModule, FFN


####模型架构
# class PatchEmbedding(nn.Module):
#     def __init__(self, batch_size=1, image_size=156, patch_size=12, in_channels=128, embed_dim=256):
#         super(PatchEmbedding, self).__init__()
#         n_patchs = (image_size // patch_size) ** 2
#         self.conv1 = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
#         # self.dropout = nn.Dropout(dropout)
#         self.class_token = torch.randn((batch_size, 1, embed_dim))
#         self.position = torch.randn((batch_size, n_patchs + 1, embed_dim))
#
#     def forward(self, x):
#         x = self.conv1(x)  # (batch,in_channel,h,w)-(batch,embed_dim,h/patch_size,w/patch_size)(1,768,14,14)
#         x = x.flatten(2)  # batch,embed_dim,h*w/(patch_size)**2   (1,768,196)
#         x = x.transpose(1, 2)  # batch,h*w/(patch_size)^^2,embed_dim  (1,196,768)
#         x = torch.concat((self.class_token, x), axis=1)  # (1,197,768)
#         x = x + self.position  # (1,197,768)
#         # x = self.dropout(x)  #(1,197,768)
#         return x

class refine(nn.Module):
    def __init__(self):
        super(refine, self).__init__()

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
        # self.conv3x1 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1, bias=False)
        # )
        self.conv1x1 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.cross = SPInterAttModule(dim=128,num_heads=1,qk_dim=128)
        self.FFN = FFN(dim=128,hidden_dim=128,out_dim=1)

    def forward_stoken(self, x, affinity_matrix):
        x = rearrange(x, 'b c h w -> b (h w) c')
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
        return stokens

    def forward(self, x, aux_data):

        global_x = self.conv(x)

        Fq = self.conv1(global_x)  # mask
        Fk = self.conv2(global_x)
        Fg = self.conv3(global_x)  # res
        # mask调整
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        aux_data = aux_data
        aux_data = F.interpolate(aux_data,size=(156,156), mode='bilinear', align_corners=False)

        Fq_mask = Fq * aux_data

        cross_attention = self.cross
        ffn = self.FFN

        attention_x = cross_attention(Fk,Fq_mask)
        attention_x = attention_x + Fg
        attention_x_res = self.conv1x1(attention_x)
        attention_x = ffn(attention_x) + attention_x_res

        result = torch.sigmoid(attention_x)
        result = F.interpolate(result, size=(640, 640), mode='bilinear', align_corners=False)

        return result





















        # elementwise_product = torch.mul(Fq, aux_data)
        # sum_result = elementwise_product.sum(dim=(2,3))
        # sum_tensor2 = aux_data.sum(dim=(2, 3))
        # fq = sum_result / (sum_tensor2+0.0001)
        #
        # ##########
        # #####中心注意力
        # # fc = self.center(Fc)
        # # fc = self.conv1x1(fc)
        # # fc = torch.sigmoid(fc)
        # # fc = F.interpolate(fc, size=(640, 640), mode='bilinear', align_corners=False)
        # # ########
        #
        #
        # # 定义线性变换层
        # # linear_layer = nn.Linear(128, 512).to('cuda')
        # #
        # # # 进行线性变换
        # # fq1 = linear_layer(fq).to('cuda')
        # #######
        #
        # # fq = torch.nan_to_num(fq, nan=0)
        # # mean_value = torch.mean(fq, dim=0)
        # # std_value = torch.std(fq, dim=0)
        # # fq = (fq - mean_value) / std_value
        # # d = torch.any(torch.isnan(fq))
        # # print(a)
        # # print(a0)
        # # print(b)
        # # print(b0)
        # # # print(c)
        # # print(c0)
        # # print(d)
        # fq = fq[:, :, None, None]
        # #####
        # # Fk_sp = p2s_assign
        # # Fk_sp = conv_layer(Fk_sp)
        # #####
        # inner_product = torch.sum(fq * Fk, dim=1)
        # result = torch.sigmoid(inner_product)
        # result = result.unsqueeze(1)
        #
        # #######
        # # fc = fc[:, :, None, None]
        # # inner_product1 = torch.sum(fc * Fk, dim=1)
        # # result1 = torch.sigmoid(inner_product1)
        # # result1 = result1.unsqueeze(1)
        # #######
        # # 两种concat
        # # result = torch.cat((result, result1), dim=1)
        # # result = self.conv1x1(result)
        # # result = torch.sigmoid(result)
        # #######
        # result = F.interpolate(result, size=(640, 640), mode='bilinear', align_corners=False)
        #
        #
        # # 反卷积
        # # conv_transpose = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=3, padding=1, output_padding=2)
        # # conv_transpose = conv_transpose.to('cuda')
        # # # 将 tensor 调整为合适的形状
        # # result = result.unsqueeze(1)
        # # # 进行反卷积操作
        # # result = conv_transpose(result, output_size=(640, 640))
        #
        #
        # # 将结果中的每个通道相加
        #
        # # has_any_nan3 = torch.any(torch.isnan(result))
        #
        # return result
        # # return result, fc
        # # return result, fq1