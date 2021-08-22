import torch

from trainer.model import DQN
from trainer.single import C, H, W, FRAME_STACKING, NUM_ACTIONS

import torch.nn as nn
import torch.nn.functional as F


class ModelWithPreprocessing(nn.Module):
    def __init__(self, frame_shape, num_actions):
        super(ModelWithPreprocessing, self).__init__()
        self.scaling_weights = torch.full(
            (12, 12, 2, 2), 0.25, dtype=torch.float)
        self.model = DQN(frame_shape, num_actions)

    def forward(self, *img_data):
        frames = torch.cat(img_data, dim=1)
        x = F.conv2d(frames, weight=self.scaling_weights, stride=2)
        x /= 255.0
        out = self.model(x)
        return out


def main():
    model = ModelWithPreprocessing((FRAME_STACKING * C, H, W), NUM_ACTIONS)
    state_dict = torch.load("../model.pt", map_location="cpu")
    model.model.load_state_dict(state_dict["model_state_dict"])

    input_names = ('frame_data_0', 'frame_data_1',
                   'frame_data_2', 'frame_data_3')

    inputs = []
    for _ in range(FRAME_STACKING):
        inputs.append(torch.randn(1, 3, 180, 240, dtype=torch.float))
    inputs = tuple(inputs)

    torch.onnx.export(
        model,
        inputs,
        "../dqn.onnx",
        input_names=input_names,
        output_names=("q_values",),
        opset_version=11,
        dynamic_axes={
            'frame_data_0': {0: 'batch'},
            'frame_data_1': {0: 'batch'},
            'frame_data_2': {0: 'batch'},
            'frame_data_3': {0: 'batch'},
            'q_values': {0: 'batch'}},
        verbose=True)


if __name__ == "__main__":
    main()
