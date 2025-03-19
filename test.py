# import re
# import matplotlib.pyplot as plt
#
#
# loss_values = []
# current_epoch = 0
#
#
# with open(
#         'outputs/workspace/DB-master/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/output.log', 'r') as file:
#     lines = file.readlines()[571:]
#     for i,line in enumerate(lines):
#         match = re.search(r'epoch: (\d+), loss: (\d+\.\d+)', line)
#         if match:
#             epoch = int(match.group(1))
#             loss = float(match.group(2))
#             loss_values.append((epoch, loss))
#             current_epoch = epoch
#
#
# if loss_values:
#     epochs, losses = zip(*loss_values)
# else:
#     epochs,losses = [], []
#
# # 创建Loss曲线
# plt.plot(epochs, losses, label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.grid()
#
# # 显示Loss曲线
# plt.show()

with open('output.txt', 'w') as file:
    for i in range(2555):
        filename = f'MPSC_img_{i}.jpg'
        file.write(f'{filename}\n')

# import torch
#
# # 假设 tensor 是一个大小为 (B, C, H, W) 的示例张量
# B, C, H, W = 16, 3, 128, 128
# tensor = torch.rand((B, C, H, W))
# print(tensor)
# # 在 H 和 W 的维度上求和
# sum_result = torch.sum(tensor, dim=(2, 3))
#
# # 打印结果
# print(sum_result)