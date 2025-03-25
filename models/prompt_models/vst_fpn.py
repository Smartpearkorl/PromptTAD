import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Tuple, Type


class MLP(nn.Module):
    """Linear Embedding."""
    def __init__(self, input_dim, embed_dim, activation: Type[nn.Module] = nn.GELU,):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = activation()

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.act(x)
        x = x.permute(0,2,1).view(b,-1,h,w).contiguous()
        return x

class DynamicAdaptiveAvgPool3d(nn.Module):
    def __init__(self):
        super(DynamicAdaptiveAvgPool3d, self).__init__()

    def forward(self, x, output_size):
        adaptive_avg_pool = nn.AdaptiveAvgPool3d(output_size)
        return adaptive_avg_pool(x)

'''
FPN
'''
class Standard_FPN(nn.Module):
    def __init__(self,
                index: list = [0,1,2,3],
                in_channels: list = [256,512,1024,1024],
                out_times = [1,1,1,1],
                embed_dims: int = [256,256,256,256,256],
                out_dims = 256,
                dimen_reduce_type: str ='mlp', # dimensionality reduction type : mlp or 1x1 conv
                activation: Type[nn.Module] = nn.GELU,):
                
        super(Standard_FPN, self).__init__()
        self.in_index = index
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.out_times = out_times
        self.out_dims = out_dims
        self.act = activation()

        self.embed_layers = nn.ModuleDict() # dimensionality reduction
        self.embed3dPooling = nn.ModuleDict() # 3D pooling
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                        self.embed_dims):
            self.embed3dPooling[str(i)] = DynamicAdaptiveAvgPool3d()
            if dimen_reduce_type =='mlp' :
                self.embed_layers[str(i)] = MLP(in_channels, embed_dim)
            elif dimen_reduce_type =='1x1':
                self.embed_layers[str(i)] = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
    
        # Smooth layers
        self.smooth = nn.Conv2d(embed_dims[-1], out_dims, kernel_size=3, stride=1, padding=1)  

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, inputs):
        # inputs: list[ [B,C,T,H,W],...]
        layer_lenth = len(inputs)
        # dimensionality reduction by mlp or 1x1 conv
        os_size = inputs[0].shape[-2:]
        for i in range(layer_lenth):
            h,w = inputs[i].shape[-2:]
            #3D adpativation
            inputs[i] = self.embed3dPooling[str(i)](inputs[i],(self.out_times[i],h,w)).squeeze(-3)
            #dimensionality reduction
            inputs[i] = self.embed_layers[str(i)](inputs[i])
        
        # Top-down
        p4 = inputs[-1]
        p3 = self._upsample_add(p4,inputs[-2])
        p2 = self._upsample_add(p3,inputs[-3])
        p1 = self._upsample_add(p2,inputs[-4])
        outputs = self.smooth(p1)
        outputs = self.act(outputs)
        return outputs

'''
FPN (Deprecated)
'''
class VST_FPN(nn.Module):
    def __init__(
            self,
            index: list = [0,1,2,3],
            in_channels: list = [256,512,1024,1024],
            out_times = [1,1,1,1],
            embed_dims: int = [256,256,256,256,256],
            # sigmoid_output: bool = False,
    ):
        super().__init__()
        self.in_index = index
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.out_times = out_times
        # self.sigmoid_output = sigmoid_output 
        self.embed_layers = nn.ModuleDict()
        self.embed3dPooling = nn.ModuleDict()
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                        self.embed_dims):
            self.embed_layers[str(i)] = MLP(in_channels, embed_dim)
            self.embed3dPooling[str(i)] = DynamicAdaptiveAvgPool3d()

    @staticmethod        
    def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
        """
        调整输入图像的大小。

        参数:
        - input: 输入的图像张量，需要至少三维（通道，高度，宽度）。
        - size: 输出图像的高度和宽度(h, w)元组。如果size为None,则根据scale_factor计算输出大小。
        - scale_factor: 用于缩放输入图像的因子。如果scale_factor为None,则根据size计算输出大小。
        - mode: 插值模式，可选值为'nearest'、'linear'、'bilinear'、'trilinear'、'area'。
        - align_corners: 是否对齐四个角的corners。仅在mode为'repeat'、'bilinear'、'trilinear'时有效。
        - warning: 是否在特定条件下发出警告。当设置为True,并且指定align_corners且size不满足特定条件时,会发出警告。

        返回:
        - 调整大小后的图像张量。
        """
        
        # 发出特定条件下的警告
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                # 如果输出高度或宽度大于输入高度或宽度，并且满足特定数学条件，则发出警告
                if output_h > input_h or output_w > output_h:
                    if ((output_h > 1 and output_w > 1 and input_h > 1
                        and input_w > 1) and (output_h - 1) % (input_h - 1)
                            and (output_w - 1) % (input_w - 1)):
                        warnings.warn(
                            f'When align_corners={align_corners}, '
                            'the output would more aligned if '
                            f'input size {(input_h, input_w)} is `x+1` and '
                            f'out size {(output_h, output_w)} is `nx+1`')
        # 调用F.interpolate进行图像大小调整
        return F.interpolate(input, size, scale_factor, mode, align_corners)

    def forward(self, inputs):
        # inputs: list[ [B,C,T,H,W],...]
        layer_lenth = len(inputs)
        outputs = []
        os_size = inputs[0].shape[-2:]
        for i in range(layer_lenth):
            h,w = inputs[i].shape[-2:]
            #3D adpativation
            inputs[i] = self.embed3dPooling[str(i)](inputs[i],(self.out_times[i],h,w)).squeeze(-3)
            inputs[i] = self.embed_layers[str(i)](inputs[i])
            if inputs[i].shape[-2:] != os_size:
                inputs[i] = self.resize(inputs[i],size=os_size)
            outputs.append(inputs[i])     
        outputs = torch.stack(outputs)
        outputs = torch.mean(outputs, dim=0)
        # if self.sigmoid_output:
        #     outputs = torch.sigmoid(outputs)
        return outputs



        