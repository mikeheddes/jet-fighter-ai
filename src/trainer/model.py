import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, frame_shape, num_actions):
        super(DQN, self).__init__()
        C, H, W = frame_shape
        self.conv1 = nn.Conv2d(
            in_channels=C, out_channels=32, kernel_size=3, padding=1)
        H1 = (H + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
        W1 = (W + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        H2 = (H1 + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
        W2 = (W1 + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
        self.fc1 = nn.Linear(32 * H2 * W2, 512)

        # Value layers
        self.vl1 = nn.Linear(512, 128)
        self.vl2 = nn.Linear(128, 1)

        # Action advantage layers
        self.al1 = nn.Linear(512, 128)
        self.al2 = nn.Linear(128, num_actions)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))

        value = F.relu(self.vl1(out))
        value = self.vl2(value)

        advantage = F.relu(self.al1(out))
        advantage = self.al2(advantage)
        mean_advantage = advantage.mean(1, keepdims=True)

        return value + advantage - mean_advantage
