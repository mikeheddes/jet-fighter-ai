import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import transforms

from trainer.env import Environment
from trainer.memory import Memory
from trainer.model import DQN
from trainer.process import Transform, Stacking, get_q_values_and_expectation

BATCH_SIZE = 28
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200
LEARNING_RATE = 0.001
FRAME_STACKING = 4
NUM_ACTIONS = 4
MEMORY_CAPACITY = 100_000

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    dqn = DQN((FRAME_STACKING, 90, 120), NUM_ACTIONS).to(device)
    optimizer = optim.RMSprop(dqn.parameters(), lr=LEARNING_RATE)
    memory = Memory(MEMORY_CAPACITY)
    toTensor = T.ToTensor()
    transform = Transform((90, 120))
    stacking = Stacking(FRAME_STACKING)

    def select_action(state, step):
        sample = random.random()
        part = 1. - min(step / EPS_DECAY, 1.)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * part
        if sample > eps_threshold:
            with torch.no_grad():
                return dqn(state).argmax().view(1, 1)
        else:
            return torch.randint(0, NUM_ACTIONS, (1, 1), device=device, dtype=torch.long)

    env = Environment()
    step = 0
    for i_episode in range(5000):
        print(f"start episode {i_episode}")

        # Get the initial observation from the game
        raw_obs = env.reset()
        stacking.reset()

        raw_obs = toTensor(raw_obs).to(device)
        transformed_obs = transform(raw_obs).unsqueeze(0)
        stacked_obs = stacking(transformed_obs)

        for step in count(step):
            action = select_action(stacked_obs, step)

            # Play an action in the game and get the next observation
            next_raw_obs, reward, done = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float, device=device)

            if done:
                next_stacked_obs = None
            else:
                next_raw_obs = toTensor(next_raw_obs).to(device)
                next_transformed_obs = transform(next_raw_obs).unsqueeze(0)
                next_stacked_obs = stacking(next_transformed_obs)

            transition = Transition(
                transformed_obs, action, reward, next_transformed_obs)
            memory.add(transition)

            stacked_obs = next_stacked_obs

            if len(memory) < 1000:
                continue

            transitions = stacking.from_memory(memory, batch_size=BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            q_values, expected_q_values = get_q_values_and_expectation(
                batch, dqn, batch_size=BATCH_SIZE, gamma=GAMMA, device=device)

            loss = F.mse_loss(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                if i_episode % 5 == 4:
                    raw_obs = env.reset()
                    stacking.reset()

                    raw_obs = toTensor(raw_obs).to(device)
                    transformed_obs = transform(raw_obs).unsqueeze(0)
                    stacked_obs = stacking(transformed_obs)

                    mean_max_q_value = 0
                    for t in count():
                        q_values = dqn(stacked_obs)
                        action = q_values.argmax().view(1, 1)
                        mean_max_q_value += q_values.max().item()

                        next_raw_obs, reward, done = env.step(action.item())
                        reward = torch.tensor(
                            [reward], dtype=torch.float, device=device)

                        if done:
                            next_stacked_obs = None
                        else:
                            next_raw_obs = toTensor(next_raw_obs).to(device)
                            next_transformed_obs = transform(
                                next_raw_obs).unsqueeze(0)
                            next_stacked_obs = stacking(next_transformed_obs)

                        transition = Transition(
                            transformed_obs, action, reward, next_transformed_obs)
                        memory.add(transition)

                        stacked_obs = next_stacked_obs

                        if done:
                            break

                    episode_duration = t + 1
                    mean_q_value = mean_max_q_value / (t + 1)
                    print("episode duration", episode_duration,
                          f"\tmean q-value: {mean_q_value:.1f}", "\tat episode", i_episode)

                if i_episode % 50 == 49:
                    torch.save({
                        'steps': step,
                        'model_state_dict': dqn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoints/model.pt")
                break


if __name__ == '__main__':
    main()
