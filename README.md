# Global-and-Dual-Attention-Mechanisms
yolov5改进Enhanced Vehicle Detection in SAR Images via Global and Dual Attention Mechanisms
模块1
参考$A^2-Nets$)，可以看做SE的进化版本



Double Attention Method
模块2
对yolov5的SPPF进行改进，在保证相同发感受野的情况下，进一步提升了模型的速度
SPPF
SPPFCSPC
模块3
参考github上别人对yolo的改进在预测头的不同层级的特征图上添加空间通道注意力**GAM_Attention**，该注意力方法可以一定
程度上解决遮挡和交叉重叠等问题
'''
class GAM_Attention(nn.Module):
def __init__(self, in_channels, out_channels, rate=4):
super(GAM_Attention, self).__init__()
self.channel_attention = nn.Sequential(
nn.Linear(in_channels, int(in_channels / rate)),
nn.ReLU(inplace=True),
nn.Linear(int(in_channels / rate), in_channels)
)
 self.spatial_attention = nn.Sequential(
nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
nn.BatchNorm2d(int(in_channels / rate)),
nn.ReLU(inplace=True),
nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
nn.BatchNorm2d(out_channels)
)
def forward(self, x):
b, c, h, w = x.shape
x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
x_channel_att = x_att_permute.permute(0, 3, 1, 2)
x = x * x_channel_att
x_spatial_att = self.spatial_attention(x).sigmoid()
out = x * x_spatial_att
return out
'''
对上述实现的简要概述： 通道注意力： 通过mlp学习一组权重与原始的输入进行加权
空间注意力： 通过卷积神经网络映射被通道加权的输入特征，并通过sigmiod激活到0-1，然后与原始的输入加权，从而使模型更
加focal感兴趣的区域
整体改进
