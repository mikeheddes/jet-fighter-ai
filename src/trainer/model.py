import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_output_size(size_in, kernel_size, padding=0, stride=1, dilation=1):
    return (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        state_size = torch.tensor(state_size, dtype=torch.int64)
        self.input_features = torch.prod(state_size)
        self.fc1 = nn.Linear(self.input_features, 24)

        # Value layers
        self.vl1 = nn.Linear(24, 12)
        self.vl2 = nn.Linear(12, 1)

        # Action advantage layers
        self.al1 = nn.Linear(24, 12)
        self.al2 = nn.Linear(12, num_actions)

    def forward(self, x):
        out = x.view(-1, self.input_features)
        out = F.relu(self.fc1(out))

        value = F.relu(self.vl1(out))
        value = self.vl2(value)

        advantage = F.relu(self.al1(out))
        advantage = self.al2(advantage)

        mean_advantage = advantage.mean(1, keepdims=True)
        return value + advantage - mean_advantage
