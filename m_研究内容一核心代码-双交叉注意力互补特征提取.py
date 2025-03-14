import torch
import torch.nn as nn


# 视觉特征提取模块（使用ResNet）
class VisionFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisionFeatureExtractor, self).__init__()
        # 第一层卷积：输入通道3，输出通道32，卷积核5x5，步长2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        # 最大池化层：池化窗口2x2
        self.conv2 = nn.MaxPool2d(2)
        # 第二层卷积：输入通道32，输出通道64，卷积核3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        # 全连接层：将特征展平后映射到128维
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18 * 18 * 64, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        # 第一层卷积 + ReLU激活
        x = self.conv1(x)
        x = torch.relu(x)
        # 最大池化
        x = self.conv2(x)
        # 第二层卷积 + ReLU激活
        x = self.conv3(x)
        x = torch.relu(x)
        # 全连接层输出
        x = self.fc(x)
        return x


# 激光雷达特征提取模块（使用一维卷积）
class LidarFeatureExtractor(nn.Module):
    def __init__(self):
        super(LidarFeatureExtractor, self).__init__()
        # 全连接层序列：用于提取激光雷达特征
        self.fc_layer = nn.Sequential(
            # 第一层全连接：将802维输入映射到128维
            nn.Linear(802, 128),
            # ReLU激活函数：引入非线性
            nn.ReLU(),
            # 第二层全连接：保持128维特征
            nn.Linear(128, 128),
            # 展平操作：将特征展平为一维
            nn.Flatten(),
            # 批归一化：加速训练并提高模型稳定性
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        # 前向传播：通过全连接层序列处理输入
        x = self.fc_layer(x)
        return x


# 双交叉注意力机制的特征增强模块
class CrossFeatureEnhancement(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(CrossFeatureEnhancement, self).__init__()
        # 注意力头数：用于多头注意力机制
        self.num_heads = num_heads
        # 可学习的线性变换矩阵：用于计算Q, K, V
        self.W_q = nn.Linear(input_dim, input_dim)  # 查询变换矩阵
        self.W_k = nn.Linear(input_dim, input_dim)  # 键变换矩阵
        self.W_v = nn.Linear(input_dim, input_dim)  # 值变换矩阵
        # 输出变换矩阵：将多头注意力结果映射回原始维度
        self.W_o = nn.Linear(input_dim, input_dim)
        # 可学习的权重参数：用于残差连接和特征融合
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 注意力结果权重
        self.beta = nn.Parameter(torch.tensor(1.0))  # 原始特征权重
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 残差连接权重
        self.delta = nn.Parameter(torch.tensor(1.0))  # 前馈网络权重
        # 前馈神经网络：用于特征增强
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),  # 扩展特征维度
            nn.ReLU(),  # 非线性激活
            nn.Linear(input_dim * 4, input_dim),  # 恢复原始维度
        )

    def forward(self, T_V, T_L):
        # 计算Q, K, V：通过线性变换得到查询、键和值
        Q_V = self.W_q(T_V)  # 视觉特征的查询向量
        K_L = self.W_k(T_L)  # 激光雷达特征的键向量
        V_L = self.W_v(T_L)  # 激光雷达特征的值向量
        
        # 多头注意力机制：将特征分割到多个注意力头
        head_dim = Q_V.size(-1) // self.num_heads  # 计算每个头的维度
        Q_V = Q_V.view(*Q_V.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        K_L = K_L.view(*K_L.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        V_L = V_L.view(*V_L.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        
        # 计算注意力分数：通过点积和缩放
        A = torch.matmul(Q_V, K_L.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32)
        )
        A = torch.softmax(A, dim=-1)  # 归一化注意力分数
        
        # 计算加权和：通过注意力分数和值向量
        Z_L = torch.matmul(A, V_L).transpose(-2, -3).contiguous()
        Z_L = Z_L.view(*Z_L.shape[:-2], -1)  # 合并多头结果
        Z_L = Z_L.view(Z_L.size(0), -1, self.W_o.in_features)  # 恢复原始形状
        Z_L = self.W_o(Z_L)  # 通过输出变换矩阵

        # 残差连接：平衡原始特征和注意力结果
        T_L_prime = self.alpha * Z_L + self.beta * T_L

        # 前馈神经网络：进一步特征增强
        T_L_hat = self.gamma * T_L_prime + self.delta * self.ffn(T_L_prime)

        return T_L_hat


# 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super(FeatureFusion, self).__init__()
        # 全连接层：将拼接后的特征映射回原始维度
        self.fc = nn.Linear(input_dim * 2, input_dim)

    def forward(self, F_V, F_L):
        # 将特征展平并拼接
        F_V_flat = F_V.view(F_V.size(0), -1)  # 展平视觉特征
        F_L_flat = F_L.view(F_L.size(0), -1)  # 展平激光雷达特征
        combined = torch.cat([F_V_flat, F_L_flat], dim=1)  # 沿特征维度拼接
        fused = torch.relu(self.fc(combined))  # 通过全连接层并激活
        return fused


# 感知模块
class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        # 视觉特征提取器：从图像中提取特征
        self.vision_extractor = VisionFeatureExtractor()
        # 激光雷达特征提取器：从点云数据中提取特征
        self.lidar_extractor = LidarFeatureExtractor()
        # 双交叉注意力机制：用于特征增强
        self.cfe = CrossFeatureEnhancement(input_dim=128)
        # 特征融合模块：将不同模态的特征进行融合
        self.fusion = FeatureFusion(input_dim=128)

    def forward(self, vision_input, lidar_input):
        # 单模态特征提取：分别处理视觉和激光雷达输入
        F_V = self.vision_extractor(vision_input)  # 提取视觉特征
        F_L = self.lidar_extractor(lidar_input.unsqueeze(1))  # 提取激光雷达特征

        # 展平特征图并添加位置嵌入
        T_V = F_V.view(F_V.size(0), -1, 128)  # 将视觉特征展平为序列
        T_L = F_L.view(F_L.size(0), -1, 128)  # 将激光雷达特征展平为序列

        # 迭代特征增强
        for _ in range(3):  # 迭代3次
            T_L = self.cfe(T_V, T_L)  # 使用视觉特征增强激光雷达特征
            T_V = self.cfe(T_L, T_V)  # 使用激光雷达特征增强视觉特征

        # 特征融合：将增强后的特征进行融合
        fused_features = self.fusion(T_V, T_L)

        return fused_features


# 示例使用
if __name__ == "__main__":
    # 生成随机视觉输入：24个样本，3通道，84x84分辨率
    vision_input = torch.randn(24, 3, 84, 84)
    # 生成随机激光雷达输入：24个样本，802维特征
    lidar_input = torch.randn(24, 802)
    # 初始化感知模块
    perception_module = PerceptionModule()
    # 前向传播：处理视觉和激光雷达输入
    output = perception_module(vision_input, lidar_input)
    # 输出特征形状
    print("Output shape:", output.shape)
