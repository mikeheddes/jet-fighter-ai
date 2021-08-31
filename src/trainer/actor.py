import torch
import random
from itertools import count
from collections import namedtuple
import gym

from trainer.model import DQN
import trainer.config as conf
from trainer.process import Transform, Stacking
# from trainer.env import Environment


# class Transition(dataobject):
#     frame: torch.FloatTensor
#     action: torch.LongTensor
#     reward: torch.FloatTensor
#     next_frame: torch.FloatTensor

#     def encode(self):
#         return (
#             (self.frame * 255.0).byte().to("cpu"),
#             self.action.to("cpu"),
#             self.reward.to("cpu"),
#             (self.next_frame * 255.0).byte().to("cpu"))

#     @classmethod
#     def decode(cls, memory_item):
#         frame, action, reward, next_frame = memory_item

#         return cls(
#             frame.float() / 255.0,
#             action,
#             reward,
#             next_frame.float() / 255.0)


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))



class Actor:
    def __init__(self, model=None, device=None):
        self.device = device
        self.step = 0
        self.env = gym.make('CartPole-v1')

        if model is None:
            self.model = DQN(
                (conf.FRAME_STACKING * conf.C, conf.H, conf.W),
                conf.NUM_ACTIONS).to(device)
        else:
            self.model = model

        # self.transform = Transform(device)
        # self.stacking = Stacking(
        #     conf.FRAME_STACKING, multi_step=conf.MULTI_STEP, gamma=conf.GAMMA)


    def policy(self, state):
        sample = random.random()
        part = 1. - min(self.step / conf.EPS_DECAY_STEPS, 1.)
        eps_threshold = conf.EPS_END + (conf.EPS_START - conf.EPS_END) * part
        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).argmax().item()
        else:
            return random.randint(0, conf.NUM_ACTIONS - 1)

    def episode(self):
        # stacking.reset()
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float)
        state = state.view(1, conf.C, conf.H, conf.W).to(self.device)
        # state = stacking(state)

        for step in count(self.step):
            self.step = step
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float)
                next_state = next_state.view(1, conf.C, conf.H, conf.W).to(self.device)
                # next_state = stacking(next_state)

            yield Transition(
                state.to("cpu"),
                action,
                reward,
                next_state.to("cpu") if next_state is not None else next_state)
        
            if done:
                return

            state = next_state



    def update_model(self, state_dict):
        self.model.load_state_dict(state_dict)
