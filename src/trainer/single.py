import os
import random
from collections import namedtuple
from itertools import count
import gc
import os
import psutil

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from trainer.env import Environment
from trainer.memory import Memory
from trainer.model import DQN
from trainer.process import Transform, Stacking, get_prediction_and_target
from trainer.types import Transition

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 5_000_000
LEARNING_RATE = 0.0005
FRAME_STACKING = 4
NUM_ACTIONS = 4
MEMORY_CAPACITY = 1_000
TARGET_NET_UPDATE_FREQ = 1000
NUM_EPISODES = 15_000
C, H, W = 3, 90, 120

def copy_model_state(from_model: torch.nn.Module, to_model: torch.nn.Module):
    state_dict = from_model.state_dict()
    to_model.load_state_dict(state_dict)


def main():
    os.makedirs("./checkpoints", exist_ok=True)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    online_dqn = DQN((FRAME_STACKING * C, H, W), NUM_ACTIONS).to(device)
    target_dqn = DQN((FRAME_STACKING * C, H, W), NUM_ACTIONS).to(device)
    copy_model_state(online_dqn, target_dqn)

    optimizer = optim.RMSprop(online_dqn.parameters(), lr=LEARNING_RATE)
    memory = Memory(MEMORY_CAPACITY)
    toTensor = T.ToTensor()
    transform = Transform((H, W))
    stacking = Stacking(FRAME_STACKING)

    def select_action(state, step):
        sample = random.random()
        part = 1. - min(step / EPS_DECAY_STEPS, 1.)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * part
        if sample > eps_threshold:
            with torch.no_grad():
                return online_dqn(state).argmax().view(1, 1)
        else:
            return torch.randint(0, NUM_ACTIONS, (1, 1), device=device, dtype=torch.long)

    env = Environment()
    step = 0

    for i_episode in range(NUM_EPISODES):
        gpu_mem = torch.cuda.memory_allocated(device) / 1e6
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1e6
        print(f"Start episode {i_episode}, using {gpu_mem:.2f} MB GPU and {cpu_mem:.2f} MB CPU")

        # Get the initial observation from the game
        raw_obs = env.reset()
        stacking.reset()

        raw_obs = toTensor(raw_obs).to(device)
        transformed_obs = transform(raw_obs).unsqueeze(0)
        stacked_obs = stacking(transformed_obs)

        for step in count(step):
            action = select_action(stacked_obs, step)

            action_repeat = random.randint(1, 4)
            for step in range(step, step + action_repeat):

                # Play an action in the game and get the next observation
                next_raw_obs, reward, done = env.step(action.item())
                reward = torch.tensor(
                    [[reward]], dtype=torch.float, device=device)

                if done:
                    next_stacked_obs = None
                else:
                    next_raw_obs = toTensor(next_raw_obs).to(device)
                    next_transformed_obs = transform(next_raw_obs).unsqueeze(0)
                    next_stacked_obs = stacking(next_transformed_obs)

                transition = Transition(
                    transformed_obs.to("cpu"),
                    action.to("cpu"),
                    reward.to("cpu"),
                    next_transformed_obs.to("cpu"))
                memory.add(transition)

                stacked_obs = next_stacked_obs

                if len(memory) < 1000:
                    continue

                transitions = stacking.from_memory(
                    memory, batch_size=BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                q_values, expected_q_values = get_prediction_and_target(
                    batch, online_dqn, target_dqn,
                    batch_size=BATCH_SIZE, gamma=GAMMA, device=device)

                loss = F.mse_loss(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % TARGET_NET_UPDATE_FREQ == (TARGET_NET_UPDATE_FREQ - 1):
                    copy_model_state(online_dqn, target_dqn)

                if done:
                    if i_episode % 5 == 4:
                        raw_obs = env.reset()
                        stacking.reset()

                        raw_obs = toTensor(raw_obs).to(device)
                        transformed_obs = transform(raw_obs).unsqueeze(0)
                        stacked_obs = stacking(transformed_obs)

                        mean_max_q_value = 0
                        for t in count():
                            q_values = online_dqn(stacked_obs)
                            action = q_values.argmax().view(1, 1)
                            mean_max_q_value += q_values.max().item()

                            next_raw_obs, reward, done = env.step(
                                action.item())
                            reward = torch.tensor(
                                [[reward]], dtype=torch.float, device=device)

                            if done:
                                next_stacked_obs = None
                            else:
                                next_raw_obs = toTensor(
                                    next_raw_obs).to(device)
                                next_transformed_obs = transform(
                                    next_raw_obs).unsqueeze(0)
                                next_stacked_obs = stacking(
                                    next_transformed_obs)

                            transition = Transition(
                                transformed_obs.to("cpu"),
                                action.to("cpu"),
                                reward.to("cpu"),
                                next_transformed_obs.to("cpu"))
                            memory.add(transition)

                            stacked_obs = next_stacked_obs

                            if done:
                                break

                        mean_q_value = mean_max_q_value / (t + 1)
                        print("episode score:", str(env.game.score),
                              f"\tmean q-value: {mean_q_value:.1f}", "\tat episode", i_episode)

                    if i_episode % 50 == 49:
                        torch.save({
                            'steps': step,
                            'model_state_dict': online_dqn.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, "checkpoints/model.pt")

                    # IMPORTANT: Stop the episode when done
                    break
            
            if done:
                break

        gc.collect()


if __name__ == '__main__':
    main()
