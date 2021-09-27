import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_output_size(size_in, kernel_size, padding=0, stride=1, dilation=1):
    return (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class LinearDQN(nn.Module):
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


class ConvDQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        C, H, W = state_size

        self.conv1 = nn.Conv2d(
            in_channels=C, out_channels=12, kernel_size=8, stride=4, padding=2)
        H = get_conv_output_size(H, 8, stride=4, padding=2)
        W = get_conv_output_size(W, 8, stride=4, padding=2)

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1)
        H = get_conv_output_size(H, 4, stride=2, padding=1)
        W = get_conv_output_size(W, 4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(
                   in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        H = get_conv_output_size(H, 3, stride=1, padding=1)
        W = get_conv_output_size(W, 3, stride=1, padding=1)

        self.linear_features = 48 * H * W
        self.fc1 = nn.Linear(self.linear_features, 24)

        # Value layers
        self.vl1 = nn.Linear(24, 12)
        self.vl2 = nn.Linear(12, 1)

        # Action advantage layers
        self.al1 = nn.Linear(24, 12)
        self.al2 = nn.Linear(12, num_actions)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = out.view(-1, self.linear_features)
        out = F.relu(self.fc1(out))

        value = F.relu(self.vl1(out))
        value = self.vl2(value)

        advantage = F.relu(self.al1(out))
        advantage = self.al2(advantage)

        mean_advantage = advantage.mean(1, keepdims=True)
        return value + advantage - mean_advantage
