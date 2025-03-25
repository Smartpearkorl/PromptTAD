import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
from dinov2.models.vision_transformer import vit_small
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14,dinov2_vitg14
# 设置补丁(patch)的高度和宽度
patch_h = 75
patch_w = 50
# 特征维度
feat_dim = 1536 # s:1536 g :1536

# 定义图像转换操作
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),  # 高斯模糊
    T.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
    T.CenterCrop((patch_h * 14, patch_w * 14)),  # 中心裁剪
    T.ToTensor(),  # 转换为张量
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
])

# 使用torch.hub加载dinov2_vits14模型并移至CUDA设备
vit_s = dinov2_vitg14(pretrained=False)
vit_s = vit_s.to('cuda')
model_path = "/home/qh/pretrain_models/dinov2_vitg14_pretrain.pth"
model_dict = torch.load(model_path, map_location="cuda")
vit_s.load_state_dict(model_dict, strict=True)

# vit_s = torch.hub.load('', 'dinov2_vits14',weight ="/home/qh/pretrain_models/dinov2_vits14_pretrain.pth", source='local').cuda()


# 创建用于存储特征和图像张量的零张量
features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()

# 图像路径
img_path = f"/home/qh/pretrain_models/dino_test.png"
# 打开图像并转换为RGB模式
img = Image.open(img_path).convert('RGB')
# 对图像进行转换操作，并将其存储在imgs_tensor的第一个位置
imgs_tensor[0] = transform(img)[:3]

# 禁用梯度计算
with torch.no_grad():
    # 将图像张量传递给dinov2_vits14模型获取特征
    features_dict = vit_s.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']
    
# 重塑特征形状为(4 * patch_h * patch_w, feat_dim)
features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()

# 创建PCA对象并拟合特征
pca = PCA(n_components=3)
pca.fit(features)

# 对PCA转换后的特征进行归一化处理
pca_features = pca.transform(features)
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())

# 根据阈值进行前景和背景的区分
pca_features_fg = pca_features[:, 0] > 0.3
pca_features_bg = ~pca_features_fg

# 查找背景特征的索引
b = np.where(pca_features_bg)

# 对前景特征再次进行PCA转换
pca.fit(features[pca_features_fg])
pca_features_rem = pca.transform(features[pca_features_fg])

# 对前景特征进行归一化处理
for i in range(3):
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    # 使用均值和标准差进行转换，个人发现这种转换方式可以得到更好的可视化效果
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

# 创建RGB特征数组
pca_features_rgb = pca_features.copy()

# 替换前景特征为转换后的特征
pca_features_rgb[pca_features_fg] = pca_features_rem

# 将背景特征设置为0
pca_features_rgb[b] = 0

# 重塑特征形状为(4, patch_h, patch_w, 3)
pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

# 显示第一个图像的RGB特征
plt.imshow(pca_features_rgb[0][...,::-1])
plt.savefig('/home/qh/pretrain_models/pca_feature.png')
plt.show()
plt.close()
