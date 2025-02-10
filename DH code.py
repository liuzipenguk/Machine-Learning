import torch
import torch.nn as nn

class RiskLimit:
    def __init__(self, risk_limits: dict, risk_lambdas: dict):
        """
        初始化 RiskLimit 模块
        参数：
            risk_limits: dict，键为风险名称（如 'delta', 'gamma', 'vega'），值为对应的风险限额（dollar term）
            risk_lambdas: dict，键为风险名称，值为对应超参数 lambda，用于控制惩罚力度
        """
        self.risk_limits = risk_limits
        self.risk_lambdas = risk_lambdas
        
    def compute_loss(self, risk_terms: dict) -> torch.Tensor:
        """
        根据输入的风险因子计算风险限制惩罚损失
        参数：
            risk_terms: dict，键为风险名称（'delta', 'gamma', 'vega'），值为当前风险值（PyTorch tensor）
        返回：
            一个标量 tensor，表示风险限制的惩罚损失
        """
        total_loss = 0.0
        for risk_name, risk_value in risk_terms.items():
            # 获取对应的风险限额和 lambda 参数
            limit = self.risk_limits.get(risk_name, None)
            lambda_val = self.risk_lambdas.get(risk_name, None)
            if limit is None or lambda_val is None:
                raise ValueError(f"风险因子 {risk_name} 对应的限额或 lambda 参数未提供。")
            
            # 计算超出部分：当 |risk_value| 超过 limit 后，超出部分才计入损失
            excess = torch.clamp(torch.abs(risk_value) - limit, min=0)
            # 对每个风险因子，计算均方惩罚损失
            penalty = lambda_val * torch.mean(excess ** 2)
            total_loss += penalty
        
        return total_loss

# 示例用法：
if __name__ == "__main__":
    # 假设你的风险因子是 delta, gamma, vega
    risk_limits = {
        "delta": 1000.0,  # 例如 delta 的风险限额是 1000 美元
        "gamma": 500.0,   # gamma 的风险限额
        "vega": 800.0     # vega 的风险限额
    }
    
    risk_lambdas = {
        "delta": 10.0,
        "gamma": 20.0,
        "vega": 15.0
    }
    
    # 创建 RiskLimit 模块实例
    risk_module = RiskLimit(risk_limits, risk_lambdas)
    
    # 假设训练过程中计算得到的风险因子（这里用随机 tensor 模拟）
    risk_terms = {
        "delta": torch.tensor([1200.0, 950.0, 1100.0]),  # 单个批次中的风险值
        "gamma": torch.tensor([600.0, 450.0, 550.0]),
        "vega":  torch.tensor([900.0, 850.0, 780.0])
    }
    
    # 计算风险限制惩罚 loss
    loss = risk_module.compute_loss(risk_terms)
    print("Risk Limit Loss:", loss.item())




################### lambda search

import torch
import torch.nn as nn
import torch.optim as optim

# 示例模型：简单全连接网络
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

# 示例损失函数
# 1. 整体PnL损失：假设用均方误差衡量模型输出与目标之间的误差
def pnl_loss_fn(output, target):
    return nn.MSELoss()(output, target)

# 2. delta风险限制损失：例如，若模型的某个风险指标（这里简单用输出的绝对值模拟）超过设定的风险限额，则惩罚
def risk_loss_fn(output, risk_limit):
    # 若 |output| 超过 risk_limit，则计算超出部分的平方损失
    violation = torch.relu(torch.abs(output) - risk_limit)
    return torch.mean(violation ** 2)

# 超参数设置
input_dim = 10
output_dim = 1
risk_limit_value = 0.5  # 例如风险限额为0.5（单位：美元等）
initial_lambda = 1.0    # 初始时对风险损失的权重
beta = 0.01             # 用于动态更新lambda的小步长
num_epochs = 100

# 创建模型和优化器
model = MyModel(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟一些数据
batch_size = 32
x = torch.randn(batch_size, input_dim)
target_pnl = torch.randn(batch_size, output_dim)  # 模拟PnL目标

# 初始化动态调整的lambda
lambda_risk = initial_lambda

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 前向传播
    output = model(x)
    
    # 分别计算两个损失项
    loss_pnl = pnl_loss_fn(output, target_pnl)
    loss_risk = risk_loss_fn(output, risk_limit_value)
    
    # 分别计算两个损失项对模型参数的梯度（不累积到 .grad 中）
    grads_pnl = torch.autograd.grad(loss_pnl, model.parameters(), retain_graph=True)
    grads_risk = torch.autograd.grad(loss_risk, model.parameters(), retain_graph=True)
    
    # 计算梯度范数（这里简单地对所有参数梯度求平方和后开根号）
    norm_pnl = sum([g.norm() ** 2 for g in grads_pnl if g is not None]) ** 0.5
    norm_risk = sum([g.norm() ** 2 for g in grads_risk if g is not None]) ** 0.5
    
    # 目标：希望 lambda * norm_risk 与 norm_pnl相当，即 lambda_target = norm_pnl / norm_risk
    lambda_target = norm_pnl / (norm_risk + 1e-8)  # 避免除0
    
    # 用滑动平均更新lambda_risk
    lambda_risk = lambda_risk + beta * (lambda_target - lambda_risk)
    
    # 组合总损失
    total_loss = loss_pnl + lambda_risk * loss_risk
    
    # 反向传播并更新参数
    total_loss.backward()
    optimizer.step()
    
    # 每10个epoch打印一次信息
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: pnl_loss = {loss_pnl.item():.4f}, risk_loss = {loss_risk.item():.4f}, "
              f"norm_pnl = {norm_pnl.item():.4f}, norm_risk = {norm_risk.item():.4f}, "
              f"lambda_target = {lambda_target.item():.4f}, lambda_risk = {lambda_risk:.4f}, "
              f"total_loss = {total_loss.item():.4f}")
        

########################Reparameterization [−risk_limit,risk_limit] ###################################

import torch
import torch.nn as nn
import torch.optim as optim

class ConstrainedHedgingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, risk_limit):
        super(ConstrainedHedgingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出一个标量delta
        self.risk_limit = risk_limit

    def forward(self, x):
        x = self.relu(self.fc1(x))
        raw_output = self.fc2(x)
        # 利用 tanh 限制输出在 (-1,1) 内，再乘以 risk_limit，确保输出在 [-risk_limit, risk_limit]
        constrained_output = self.risk_limit * torch.tanh(raw_output)
        return constrained_output

# 示例设置
input_dim = 10
hidden_dim = 20
risk_limit = 0.5  # 例如：风险限额为0.5

model = ConstrainedHedgingModel(input_dim, hidden_dim, risk_limit)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义原始的PnL损失函数（比如均方误差）
def pnl_loss_fn(pred, target):
    return nn.MSELoss()(pred, target)

# 模拟训练
for epoch in range(100):
    optimizer.zero_grad()
    x = torch.randn(32, input_dim)
    target = torch.randn(32, 1)  # 模拟的PnL目标
    output = model(x)
    loss = pnl_loss_fn(output, target)
    loss.backward()
    optimizer.step()

    # 检查输出是否满足约束
    if epoch % 10 == 0:
        max_delta = output.abs().max().item()
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, max(|delta|): {max_delta:.4f}")



########################小范围改动模型#################################

def project_to_constraint(output, risk_limit):
    # 直接将输出剪裁到 [-risk_limit, risk_limit]
    return torch.clamp(output, -risk_limit, risk_limit)

# 在训练循环中：
output = model(x)
# 投影使得输出满足约束
output = project_to_constraint(output, risk_limit)
loss = pnl_loss_fn(output, target)


############### normalization #######################

import numpy as np
from sklearn.preprocessing import StandardScaler

# 假设我们有一些训练数据和测试数据
train_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
test_data = np.array([[5, 6], [6, 7]])

# 初始化StandardScaler
scaler = StandardScaler()

# 使用训练数据拟合scaler（计算均值和标准差）
scaler.fit(train_data)

# 对训练数据进行归一化
train_data_normalized = scaler.transform(train_data)

# 对测试数据进行归一化（使用训练数据的均值和标准差）
test_data_normalized = scaler.transform(test_data)

print("归一化后的训练数据:")
print(train_data_normalized)

print("归一化后的测试数据:")
print(test_data_normalized)

# 4. 对训练集和测试集进行归一化
train_normalized = scaler.transform(train_data)
test_normalized = scaler.transform(test_data)  # 关键：测试集用训练集的参数！

import joblib
joblib.dump(scaler, 'scaler.pkl')  # 保存

# 加载使用
loaded_scaler = joblib.load('scaler.pkl')
new_data_normalized = loaded_scaler.transform(new_raw_data)