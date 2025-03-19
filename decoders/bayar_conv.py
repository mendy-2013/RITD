import torch
import torch.nn  as nn
import torch.nn.functional as F

class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Differential_block(nn.Module):
    # Differential_block with 2 convolutions
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, padding = None, act = True, e = 2):
        super(Differential_block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mid = int(out_channels // e)

        # first convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        # BayarConv
        self.bayar_conv = BayarConv2d(out_channels, self.mid)

        # second convolution
        self.conv2 = nn.Conv2d(self.mid, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        identify = x        # residual 
        x = self.bayar_conv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # return  torch.cat((x,identify),dim=1)    cat
        return  identify + x        # add

if __name__ == "__main__":
    input_x = torch.randn((1,2,32,32))
    # model = BayarConv2d(3,1)
    model = Differential_block(2,2)
    print(model)
    output_x = model(input_x)
    print(output_x.shape)
