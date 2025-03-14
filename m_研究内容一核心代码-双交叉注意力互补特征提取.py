import torch
import torch.nn as nn


# 视觉特征提取模块（使用ResNet简化版本）
class VisionFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisionFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18 * 18 * 64, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x


# 激光雷达特征提取模块（使用一维卷积）
class LidarFeatureExtractor(nn.Module):
    def __init__(self):
        super(LidarFeatureExtractor, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(802, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Flatten(),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x


# 双交叉注意力机制的特征增强模块
class CrossFeatureEnhancement(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(CrossFeatureEnhancement, self).__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim),
        )

    def forward(self, T_V, T_L):
        # 计算Q, K, V
        Q_V = self.W_q(T_V)
        K_L = self.W_k(T_L)
        V_L = self.W_v(T_L)
        # 多头注意力机制
        head_dim = Q_V.size(-1) // self.num_heads
        Q_V = Q_V.view(*Q_V.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        K_L = K_L.view(*K_L.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        V_L = V_L.view(*V_L.shape[:-1], self.num_heads, head_dim).transpose(-2, -3)
        A = torch.matmul(Q_V, K_L.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32)
        )
        A = torch.softmax(A, dim=-1)
        Z_L = torch.matmul(A, V_L).transpose(-2, -3).contiguous()
        Z_L = Z_L.view(*Z_L.shape[:-2], -1)  # 先将多头的结果合并
        Z_L = Z_L.view(Z_L.size(0), -1, self.W_o.in_features)  # 恢复原始形状
        Z_L = self.W_o(Z_L)

        # 残差连接
        T_L_prime = self.alpha * Z_L + self.beta * T_L

        # 前馈神经网络
        T_L_hat = self.gamma * T_L_prime + self.delta * self.ffn(T_L_prime)

        return T_L_hat


# 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super(FeatureFusion, self).__init__()
        self.fc = nn.Linear(input_dim * 2, input_dim)

    def forward(self, F_V, F_L):
        # 将特征展平并拼接
        F_V_flat = F_V.view(F_V.size(0), -1)
        F_L_flat = F_L.view(F_L.size(0), -1)
        combined = torch.cat([F_V_flat, F_L_flat], dim=1)
        fused = torch.relu(self.fc(combined))
        return fused


# 感知模块
class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.vision_extractor = VisionFeatureExtractor()
        self.lidar_extractor = LidarFeatureExtractor()
        self.cfe = CrossFeatureEnhancement(input_dim=128)
        self.fusion = FeatureFusion(input_dim=128)

    def forward(self, vision_input, lidar_input):
        # 单模态特征提取
        F_V = self.vision_extractor(vision_input)
        F_L = self.lidar_extractor(lidar_input.unsqueeze(1))

        # 展平特征图并添加位置嵌入（简化实现，这里省略位置嵌入）
        T_V = F_V.view(F_V.size(0), -1, 128)
        T_L = F_L.view(F_L.size(0), -1, 128)

        # 特征增强
        T_L_hat = self.cfe(T_V, T_L)
        T_V_hat = self.cfe(T_L, T_V)

        # 特征融合
        fused_features = self.fusion(T_V_hat, T_L_hat)

        return fused_features


# 示例使用
if __name__ == "__main__":
    vision_input = torch.randn(24, 3, 84, 84)
    lidar_input = torch.randn(24, 802)
    perception_module = PerceptionModule()
    output = perception_module(vision_input, lidar_input)
    print("Output shape:", output.shape)
