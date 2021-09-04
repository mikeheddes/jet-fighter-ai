import torch
import torch.nn as nn

from trainer.model import DQN
from trainer.single import C, H, W, FRAME_STACKING, NUM_ACTIONS
from trainer.process import Transform



class ModelWithPreprocessing(nn.Module):
    def __init__(self, frame_shape, num_actions):
        super().__init__()
        self.transform = Transform(channels=C)
        self.model = DQN(frame_shape, num_actions)

    def forward(self, *img_data):
        img_data = [self.transform(im) for im in img_data]
        x = torch.cat(img_data, dim=1)
        x /= 255.0
        out = self.model(x)
        return out


def main():
    model = ModelWithPreprocessing((FRAME_STACKING * C, H, W), NUM_ACTIONS)
    state_dict = torch.load("../model-4.pt", map_location="cpu")
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
