import torch.nn as nn
import torch.nn.functional as F


def get_conv_output_size(size_in, kernel_size, padding=0, stride=1, dilation=1):
    return (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):
    def __init__(self, frame_shape, num_actions):
        super(DQN, self).__init__()
        C, H, W = frame_shape

        # self.conv1 = nn.Conv2d(
        #     in_channels=C, out_channels=16, kernel_size=8, stride=4, padding=2)
        # H = get_conv_output_size(H, 8, stride=4, padding=2)
        # W = get_conv_output_size(W, 8, stride=4, padding=2)

        # self.conv2 = nn.Conv2d(
        #     in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        # H = get_conv_output_size(H, 4, stride=2, padding=1)
        # W = get_conv_output_size(W, 4, stride=2, padding=1)
        self.num_conv2_features = C * H * W

        # self.fc1 = nn.Linear(32 * H * W, 512)
        self.fc1 = nn.Linear(C * H * W, 48)

        # Value layers
        self.vl1 = nn.Linear(48, 12)
        self.vl2 = nn.Linear(12, 1)

        # Action advantage layers
        self.al1 = nn.Linear(48, 12)
        self.al2 = nn.Linear(12, num_actions)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        # out = F.relu(self.conv2(out))
        out = x.view(-1, self.num_conv2_features)
        out = F.relu(self.fc1(out))

        value = F.relu(self.vl1(out))
        value = self.vl2(value)

        advantage = F.relu(self.al1(out))
        advantage = self.al2(advantage)
        mean_advantage = advantage.mean(1, keepdims=True)

        return value + advantage - mean_advantage
