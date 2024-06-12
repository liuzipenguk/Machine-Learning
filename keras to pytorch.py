import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyLayer(nn.Module):
    def __init__(self, d=None, m=None, use_batch_norm=None, 
                 kernel_initializer="he_uniform", activation_dense="relu", 
                 activation_output="linear", delta_constraint=None, day=None):
        super(StrategyLayer, self).__init__()
        self.d = d
        self.m = m
        self.use_batch_norm = use_batch_norm
        self.activation_dense = activation_dense
        self.activation_output = activation_output
        self.delta_constraint = delta_constraint
        
        self.intermediate_dense = nn.ModuleList()
        self.intermediate_BN = nn.ModuleList()

        for i in range(d):
            dense = nn.Linear(m if i > 0 else 1, m)
            self._initialize_weights(dense, kernel_initializer)
            self.intermediate_dense.append(dense)
            
            if self.use_batch_norm:
                self.intermediate_BN.append(nn.BatchNorm1d(m))
        
        self.output_dense = nn.Linear(m, 1)
        self._initialize_weights(self.output_dense, kernel_initializer)
        
    def _initialize_weights(self, layer, initializer):
        if initializer == "he_normal":
            nn.init.kaiming_normal_(layer.weight)
        elif initializer == "he_uniform":
            nn.init.kaiming_uniform_(layer.weight)
        elif initializer == "truncated_normal":
            nn.init.trunc_normal_(layer.weight, std=0.02)
        else:
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        for i in range(self.d):
            x = self.intermediate_dense[i](x)
            
            if self.use_batch_norm:
                x = self.intermediate_BN[i](x)
                
            if self.activation_dense == "leaky_relu":
                x = F.leaky_relu(x)
            else:
                x = getattr(F, self.activation_dense)(x)
         
        x = self.output_dense(x)
					 
        if self.activation_output == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.activation_output in ["sigmoid", "tanh", "hard_sigmoid"]:
            x = getattr(torch, self.activation_output)(x)
            if self.delta_constraint is not None:
                delta_min, delta_max = self.delta_constraint
                x = (delta_max - delta_min) * x + delta_min
        return x
    
class DeepHedgingModel(nn.Module):
    def __init__(self, N=None, d=None, m=None, risk_free=None, dt=None, 
                 initial_wealth=0.0, epsilon=0.0, final_period_cost=False, 
                 strategy_type=None, use_batch_norm=None, kernel_initializer="he_uniform", 
                 activation_dense="relu", activation_output="linear", 
                 delta_constraint=None, share_strategy_across_time=False, 
                 cost_structure="proportional"):
        super(DeepHedgingModel, self).__init__()
        self.N = N
        self.d = d
        self.m = m
        self.risk_free = risk_free
        self.dt = dt
        self.initial_wealth = initial_wealth
        self.epsilon = epsilon
        self.final_period_cost = final_period_cost
        self.strategy_type = strategy_type
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = kernel_initializer
        self.activation_dense = activation_dense
        self.activation_output = activation_output
        self.delta_constraint = delta_constraint
        self.share_strategy_across_time = share_strategy_across_time
        self.cost_structure = cost_structure
        
        self.strategy_layers = nn.ModuleList()
        for j in range(N):
            if not share_strategy_across_time or j == 0:
                layer = StrategyLayer(d, m, use_batch_norm, kernel_initializer, 
                                      activation_dense, activation_output, 
                                      delta_constraint, day=j)
                self.strategy_layers.append(layer)
        
    def forward(self, inputs):
        prc, information_set, *extra_inputs = inputs
        wealth = self.initial_wealth
        
        for j in range(self.N):
            if self.strategy_type == "simple":
                helper1 = information_set
            elif self.strategy_type == "recurrent":
                if j == 0:
                    strategy = torch.zeros_like(prc)
                helper1 = torch.cat([information_set, strategy], dim=-1)

            if self.share_strategy_across_time:
                strategy_layer = self.strategy_layers[0]
            else:
                strategy_layer = self.strategy_layers[j]

            strategy = strategy_layer(helper1)
            
            if j == 0:
                delta_strategy = strategy
            else:
                delta_strategy = strategy - prev_strategy
            
            if self.cost_structure == "proportional":
                costs = self.epsilon * torch.abs(delta_strategy) * prc
            elif self.cost_structure == "constant":
                costs = self.epsilon

            wealth -= costs.sum(dim=1, keepdim=True)
            wealth -= (delta_strategy * prc).sum(dim=1, keepdim=True)
            wealth *= torch.exp(self.risk_free * self.dt)
            
            prev_strategy = strategy
            prc, information_set = extra_inputs[2*j:2*j+2]
        
        if self.final_period_cost:
            if self.cost_structure == "proportional":
                costs = self.epsilon * torch.abs(prev_strategy) * prc
            elif self.cost_structure == "constant":
                costs = self.epsilon
            wealth -= costs.sum(dim=1, keepdim=True)
        
        wealth += (prev_strategy * prc).sum(dim=1, keepdim=True)
        payoff = extra_inputs[-1]
        wealth += payoff
        
        return wealth

def delta_sub_model(model, days_from_today, share_strategy_across_time=False, strategy_type="simple"):
    if strategy_type == "simple":
        inputs = torch.zeros(1)
        intermediate_inputs = inputs
    elif strategy_type == "recurrent":
        inputs = [torch.zeros(1), torch.zeros(1)]
        intermediate_inputs = torch.cat(inputs, dim=-1)
        
    if not share_strategy_across_time:
        outputs = model.strategy_layers[days_from_today](intermediate_inputs)
    else:
        outputs = model.strategy_layers[0](intermediate_inputs)
        
    return nn.Sequential(nn.Identity(), outputs)

# Usage Example
model = DeepHedgingModel(
    N=10, 
    d=3, 
    m=64, 
    risk_free=0.01, 
    dt=1/252, 
    initial_wealth=1000, 
    epsilon=0.01, 
    final_period_cost=True, 
    strategy_type="simple", 
    use_batch_norm=True, 
    kernel_initializer="he_uniform", 
    activation_dense="relu", 
    activation_output="sigmoid", 
    delta_constraint=(0, 1), 
    share_strategy_across_time=False, 
    cost_structure="proportional"
)

# This is a simple dummy input to test the model
inputs = [torch.randn(1, 1) for _ in range(22)]  # Adjust the number of inputs accordingly
outputs = model(inputs)

print(outputs)












################
# simplified


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TradeWealthModel(nn.Module):
    def __init__(self, m=1, d=1, N=20):
        super(TradeWealthModel, self).__init__()
        self.m = m
        self.d = d
        self.N = N

        self.layers = nn.ModuleList()
        for j in range(N):
            for i in range(d):
                layer = nn.Linear(2 * m, m)
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
                self.layers.append(layer)

    def forward(self, price, strategy, wealth, costs, incr_list):
        for j in range(self.N):
            helper1 = torch.cat((price, strategy), dim=1)
            for i in range(self.d):
                strategy_helper = torch.tanh(self.layers[i + j * self.d](helper1))

            change = strategy_helper - strategy
            strategy = strategy_helper
            absolute_changes = torch.abs(change)
            costs = -0.01 * torch.sum(absolute_changes * price, dim=1, keepdim=True)

            price = price + incr_list[j]
            mult = torch.sum(strategy_helper * incr_list[j], dim=1, keepdim=True)
            wealth = wealth + mult + costs

        helper = torch.abs(strategy)
        costs = -0.01 * torch.sum(helper * price, dim=1, keepdim=True)
        wealth = wealth + costs

        return wealth

# Custom loss function
def custom_loss(y_pred):
    return torch.mean(torch.exp(-y_pred))

# Hyperparameters
m = 1  # dimension of state space
d = 1  # number of layers in strategy
N = 20  # time discretization
Ktrain = 10**4
initialprice = 1.0
initialwealth = 1.0

# Create model
model = TradeWealthModel(m, d, N)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data
price = torch.tensor(initialprice * np.ones((Ktrain, m)), dtype=torch.float32)
strategy = torch.tensor(0.0 * np.ones((Ktrain, m)), dtype=torch.float32)
wealth = torch.tensor(initialwealth * np.ones((Ktrain, 1)), dtype=torch.float32)
costs = torch.tensor(np.zeros((Ktrain, 1)), dtype=torch.float32)
incr_list = [torch.tensor(np.random.normal(0.1 / N, 0.2 / np.sqrt(N), (Ktrain, m)), dtype=torch.float32) for i in range(N)]

xtrain = [price, strategy, costs, wealth] + incr_list
ytrain = torch.zeros((Ktrain, 1), dtype=torch.float32)

# Training loop
num_epochs = 100  # Example number of epochs

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(price, strategy, wealth, costs, incr_list)

    # Compute loss
    loss = custom_loss(output)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
