import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class VisualEncoder(nn.Module):
    """视觉编码器模块，用于处理视觉输入
    结构包含3层CNN和2层全连接层
    输入: 视觉观测数据(通常是图像)
    输出: 编码后的特征向量
    """

    def __init__(self, obs_shape, output_dim):
        """初始化视觉编码器
        Args:
            obs_shape: 输入观测的形状 (channels, height, width)
            output_dim: 输出特征的维度
        """
        super().__init__()
        # 3层卷积网络结构
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 8, 4)  # 第一层卷积: 32个8x8卷积核，步长4
        self.conv2 = nn.Conv2d(32, 64, 4, 2)  # 第二层卷积: 64个4x4卷积核，步长2
        self.conv3 = nn.Conv2d(64, 64, 3, 1)  # 第三层卷积: 64个3x3卷积核，步长1

        # 计算卷积层输出尺寸
        conv_out_size = self._get_conv_output(obs_shape)

        # 全连接层结构
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # 第一层全连接
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(128, output_dim),  # 第二层全连接
            nn.ReLU()  # ReLU激活函数
        )

        # 初始化卷积层权重
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))  # 使用正交初始化

    def _get_conv_output(self, shape):
        """计算卷积层输出的特征维度大小
        Args:
            shape: 输入张量的形状 (channels, height, width)
        Returns:
            int: 展平后的特征维度总数
        """
        with torch.no_grad():  # 禁用梯度计算
            dummy = torch.zeros(1, *shape)  # 创建全零的假输入张量
            # 通过所有卷积层前向传播
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            return int(np.prod(dummy.size()))  # 计算并返回展平后的特征总数

    def forward(self, x):
        """前向传播过程
        Args:
            x: 输入张量，可以是3维(单帧)或4维(批量)
        Returns:
            编码后的特征向量
        """
        if x.ndim == 3:  # 如果是单帧输入(无batch维度)
            x = x.unsqueeze(0)  # 添加batch维度
        # 通过三层卷积网络，每层后接ReLU激活
        x = F.relu(self.conv1(x))  # 第一层卷积+激活
        x = F.relu(self.conv2(x))  # 第二层卷积+激活
        x = F.relu(self.conv3(x))  # 第三层卷积+激活
        x = x.view(x.size(0), -1)  # 展平特征图(保留batch维度)
        return self.fc(x)  # 通过全连接层输出最终特征


# 激光雷达特征提取模块（使用一维卷积）
class LidarFeatureExtractor(nn.Module):
    def __init__(self, obs_shape, output_dim):
        super(LidarFeatureExtractor, self).__init__()
        # 全连接层序列：用于提取激光雷达特征
        self.fc_layer = nn.Sequential(
            # 第一层全连接：将802维输入映射到128维
            nn.Linear(obs_shape[0], output_dim),
            # ReLU激活函数：引入非线性
            nn.ReLU(),
            # 第二层全连接：保持128维特征
            nn.Linear(output_dim, output_dim),
            # 展平操作：将特征展平为一维
            nn.Flatten(),
            # 批归一化：加速训练并提高模型稳定性
            nn.BatchNorm1d(output_dim),
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


# 多尺度特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super(FeatureFusion, self).__init__()
        # 多尺度特征提取网络
        self.multi_scale = nn.Sequential(
            nn.Conv1d(input_dim * 2, 64, kernel_size=3, padding=1),  # 局部特征提取
            nn.Conv1d(input_dim * 2, 64, kernel_size=5, padding=2),  # 大范围特征提取
            nn.AdaptiveAvgPool1d(1)  # 全局特征提取
        )
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, input_dim)
        )
        # 残差连接
        self.res_conv = nn.Linear(input_dim * 2, input_dim)

    def forward(self, F_V, F_L):
        # 特征拼接
        combined = torch.cat([F_V, F_L], dim=2).permute(0, 2, 1)  # [B, 256, N]

        # 多尺度特征提取
        scale1 = self.multi_scale[0](combined).mean(dim=-1)  # 局部特征
        scale2 = self.multi_scale[1](combined).mean(dim=-1)  # 全局特征
        scale3 = self.multi_scale[2](combined).squeeze(-1)  # 上下文特征

        # 多尺度融合
        fused = torch.cat([scale1, scale2, scale3], dim=1)
        fused = self.fusion_layer(fused)

        # 残差连接
        residual = self.res_conv(torch.cat([F_V.mean(1), F_L.mean(1)], dim=1))
        return torch.relu(fused + residual)


class VectorEncoder(nn.Module):
    """处理低维向量输入的直通编码器"""

    def __init__(self, obs_shape):
        super().__init__()
        self.output_size = obs_shape[0]

    def forward(self, x):
        return x  # 直接返回原始特征


class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        super().__init__()
        self.observation_space_shape = observation_space
        self.fusion_output_size = 128
        encoder_output_size = self.fusion_output_size
        for i, obs_shape in enumerate(self.observation_space_shape):
            if len(obs_shape) > 1:
                # 视觉特征提取器：从图像中提取特征
                self.vision_extractor = VisualEncoder(obs_shape, output_dim=self.fusion_output_size)
                # self.vision_extractor = VisionFeatureExtractorResNet(obs_shape, output_dim=self.fusion_output_size)
            elif obs_shape[0] > 50:
                # 激光雷达特征提取器：从点云数据中提取特征
                self.lidar_extractor = LidarFeatureExtractor(obs_shape, output_dim=self.fusion_output_size)
            else:
                # 向量编码器：处理低维向量输入
                self.vector_encoder = VectorEncoder(obs_shape)
                encoder_output_size += obs_shape[0]

        # 双交叉注意力机制：用于特征增强
        self.cfe = CrossFeatureEnhancement(input_dim=self.fusion_output_size)
        # 特征融合模块：将不同模态的特征进行融合
        self.fusion = FeatureFusion(input_dim=self.fusion_output_size)

        # RNN配置参数
        self.recurrence = config["recurrence"]

        # 循环层（保持原有结构）
        self.recurrent_layer = self._build_recurrent_layer(
            config["recurrence"],
            encoder_output_size
        )

        # 后续网络结构（保持原有结构）
        self._build_policy_value_networks(
            config["hidden_layer_size"],
            action_space_shape
        )

    def _build_recurrent_layer(self, recurrence_cfg, input_size):
        """构建并初始化循环神经网络层

        Args:
            recurrence_cfg (dict): 循环网络配置字典，包含:
                - layer_type: 网络类型('gru'或'lstm')
                - hidden_state_size: 隐藏层大小
            input_size (int): 输入特征维度

        Returns:
            nn.Module: 初始化好的循环神经网络层(GRU或LSTM)
        """
        # 根据配置选择GRU或LSTM
        if recurrence_cfg["layer_type"] == "gru":
            layer = nn.GRU(input_size, recurrence_cfg["hidden_state_size"], batch_first=True)
        else:
            layer = nn.LSTM(input_size, recurrence_cfg["hidden_state_size"], batch_first=True)

        # 初始化网络参数
        for name, param in layer.named_parameters():
            if "bias" in name:  # 偏置项初始化为0
                nn.init.constant_(param, 0)
            elif "weight" in name:  # 权重使用正交初始化
                nn.init.orthogonal_(param, np.sqrt(2))
        return layer

    def _build_policy_value_networks(self, hidden_size, action_size):
        """构建策略网络和价值网络

        Args:
            hidden_size (int): 隐藏层维度大小
            action_size (int): 动作空间维度大小
        """
        # 共享隐藏层: 将RNN输出映射到隐藏空间
        self.lin_hidden = nn.Linear(
            self.recurrent_layer.hidden_size,
            hidden_size
        )
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))  # 正交初始化

        # 策略分支: 输出动作分布参数
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 隐藏层
            nn.ReLU(),  # 非线性激活
            nn.Linear(hidden_size, action_size)  # 输出层
        )
        nn.init.orthogonal_(self.policy_head[-1].weight, np.sqrt(2))  # 正交初始化

        # 价值分支: 输出状态价值估计
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 隐藏层
            nn.ReLU(),  # 非线性激活
            nn.Linear(hidden_size, 1)  # 输出单个价值估计
        )
        nn.init.orthogonal_(self.value_head[-1].weight, 1)  # 正交初始化(缩放因子1)

        # 标准差参数: 用于连续动作空间的动作探索
        self.log_std = nn.Parameter(torch.zeros(1, action_size))  # 初始化为0

    def forward(self, obs, recurrent_cell, device, seq_len=1):
        """Actor-Critic模型的前向传播过程

        Args:
            obs: 多模态观测输入列表
            recurrent_cell: RNN/LSTM的隐藏状态
            device: 计算设备(cpu/gpu)
            seq_len: 序列长度(用于处理时序数据)

        Returns:
            mean: 动作均值
            std: 动作标准差
            value: 状态价值估计
            recurrent_cell: 更新后的RNN隐藏状态
        """
        # 1. 多模态输入预处理
        vision_input, lidar_input, vector_input = None, None, None
        for i, obs_shape in enumerate(self.observation_space_shape):
            # 处理每个模态的输入数据
            if not isinstance(obs[i], torch.Tensor):
                h_i = torch.tensor(obs[i], dtype=torch.float32, device=device)
            else:
                h_i = obs[i].clone().detach()
            h_i.to(device)

            # 根据输入形状分发到不同模态处理器
            if len(obs_shape) > 1:  # 视觉输入
                if len(h_i.shape) == 3:
                    h_i = h_i.unsqueeze(0)  # 添加batch维度
                vision_input = h_i
            elif obs_shape[0] > 50:  # 激光雷达输入
                lidar_input = h_i
            else:  # 向量输入
                vector_input = h_i

        # 2. 特征提取阶段
        F_V = self.vision_extractor(vision_input)  # 视觉特征提取
        F_L = self.lidar_extractor(lidar_input.unsqueeze(1))  # 激光雷达特征提取
        F_Vec = self.vector_encoder(vector_input)  # 向量特征提取

        # 3. 特征增强与融合
        # 将特征展平为序列形式
        T_V = F_V.view(F_V.size(0), -1, self.fusion_output_size)
        T_L = F_L.view(F_L.size(0), -1, self.fusion_output_size)

        # 双交叉注意力特征增强(迭代3次)
        for _ in range(3):
            T_L = self.cfe(T_V, T_L)  # 视觉→激光雷达特征增强
            T_V = self.cfe(T_L, T_V)  # 激光雷达→视觉特征增强

        # 多模态特征融合
        fused_features = self.fusion(T_V, T_L)

        # 4. 特征合并与循环网络处理
        h = torch.cat([fused_features, F_Vec], dim=1)  # 合并所有特征

        # 处理序列数据
        if seq_len > 1:
            batch_size = h.size(0) // seq_len
            h = h.view(batch_size, seq_len, -1)  # 按序列长度重组
        else:
            h = h.unsqueeze(1)  # 添加序列维度

        # 通过循环网络处理时序特征
        h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)
        h = h.reshape(-1, self.recurrent_layer.hidden_size)

        # 5. 输出处理
        h = F.relu(self.lin_hidden(h))  # 共享隐藏层

        # 策略分支输出(动作分布)
        mean = torch.tanh(self.policy_head(h))  # 动作均值(-1,1范围)
        # 价值分支输出(状态价值)
        value = self.value_head(h).squeeze(-1)
        # 动作标准差(固定参数)
        std = self.log_std.exp().clamp(1e-5, 10).expand_as(mean)

        return mean, std, value, recurrent_cell

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        """初始化循环神经网络的隐藏状态和cell状态(全零初始化)

        Args:
            num_sequences (int): 需要初始化的序列数量
            device (torch.device): 计算设备(cpu/gpu)

        Returns:
            tuple: 包含两个元素的元组:
                - hxs: 初始化的隐藏状态
                - cxs: LSTM特有的cell状态(GRU时为None)
        """
        # 初始化隐藏状态(全零张量)
        hxs = torch.zeros(
            (num_sequences),
            self.recurrence["hidden_state_size"],  # 隐藏层大小
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)  # 添加序列维度

        cxs = None  # 默认无cell状态(GRU情况)

        # 如果是LSTM则初始化cell状态
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros(
                (num_sequences),
                self.recurrence["hidden_state_size"],
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

        return hxs, cxs
