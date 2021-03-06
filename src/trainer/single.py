import os
import random
from itertools import count
import gc
import os
import psutil
import time
import gym

import torch
import torch.optim as optim
import torchvision.transforms as T

from trainer.env import Environment
# from trainer.prioritized_memory import Memory
from trainer.memory import Memory
from trainer.model import DQN
from trainer.process import Transform, Stacking, get_prediction_and_target
from trainer.types import Transition, TransitionBatch

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 5_000_000
LEARNING_RATE = 0.00025
FRAME_STACKING = 4
# NUM_ACTIONS = 4
NUM_ACTIONS = 2
MEMORY_CAPACITY = 500_000
MIN_MEMORY_SIZE = 10_000
TARGET_NET_UPDATE_FREQ = 10_000
NUM_EPISODES = 15_000
# C, H, W = 1, 90, 120
C, H, W = 1, 1, 4
MULTI_STEP = 3


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
    loss_fn = torch.nn.MSELoss() # reduction='none')
    memory = Memory(MEMORY_CAPACITY)
    toTensor = T.ToTensor()
    transform = Transform(C)
    stacking = Stacking(FRAME_STACKING, multi_step=MULTI_STEP, gamma=GAMMA)
    env = gym.make('CartPole-v1')

    global policy_invocations
    policy_invocations = 0

    def select_action(state):
        global policy_invocations
        sample = random.random()
        part = 1. - min(policy_invocations / EPS_DECAY_STEPS, 1.)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * part
        policy_invocations += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return online_dqn(state).argmax().view(1, 1)
        else:
            return torch.randint(0, NUM_ACTIONS, (1, 1), device=device, dtype=torch.long)


    def optimize_model():
        if len(memory) < MIN_MEMORY_SIZE:
            return

        transitions, sample_ids, is_weights = stacking.from_memory(
            memory, batch_size=BATCH_SIZE)
        batch = TransitionBatch(*zip(*transitions))

        pred, target = get_prediction_and_target(
            batch, online_dqn, target_dqn,
            batch_size=BATCH_SIZE, gamma=GAMMA,
            multi_step=MULTI_STEP, device=device)

        errors = torch.abs(pred - target)
        for batch_i in range(BATCH_SIZE):
            memory.update_priority(
                sample_ids[batch_i],
                errors[batch_i].item())

        # is_weights = is_weights.to(device)
        # losses = loss_fn(pred, target) * is_weights
        # loss = losses.mean()

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def rollout():
        raw_obs = env.reset()
        stacking.reset()

        # raw_obs = toTensor(raw_obs).to(device)
        # raw_obs = raw_obs.unsqueeze(0)
        # transformed_obs = transform(raw_obs)
        transformed_obs = torch.tensor(
            raw_obs, dtype=torch.float).view(1, 1, 1, 4).to(device)
        stacked_obs = stacking(transformed_obs)

        mean_max_q_value = 0
        rewards = 0
        for t in count():
            q_values = online_dqn(stacked_obs)
            action = q_values.argmax().view(1, 1)
            mean_max_q_value += q_values.max().item()

            next_raw_obs, reward, done, _ = env.step(
                action.item())
            rewards += reward
            reward = torch.tensor(
                [[reward]], dtype=torch.float, device=device)

            if done:
                next_stacked_obs = None
            else:
                # next_raw_obs = toTensor(next_raw_obs).to(device)
                # next_raw_obs = next_raw_obs.unsqueeze(0)
                # next_transformed_obs = transform(next_raw_obs)
                next_transformed_obs = torch.tensor(
                    next_raw_obs, dtype=torch.float).view(1, 1, 1, 4).to(device)
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
        print("episode score:", str(rewards),
                f"\tmean q-value: {mean_q_value:.4g}",
                "\tat episode", i_episode)

    step = 0
    done = False

    for i_episode in range(NUM_EPISODES):
        gpu_mem = torch.cuda.memory_allocated(device) / 1e6
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1e6
        print(
            f"Start episode {i_episode}, using {gpu_mem:.2f} MB GPU and {cpu_mem:.2f} MB CPU")

        # Get the initial observation from the game
        raw_obs = env.reset()
        stacking.reset()

        # raw_obs = toTensor(raw_obs).to(device)
        # raw_obs = raw_obs.unsqueeze(0)
        # transformed_obs = transform(raw_obs)
        transformed_obs = torch.tensor(
            raw_obs, dtype=torch.float).view(1, 1, 1, 4).to(device)
        stacked_obs = stacking(transformed_obs)

        for step in count(step):
            action = select_action(stacked_obs)

            action_repeat = random.randint(1, 1)
            for step in range(step, step + action_repeat):

                # Play an action in the game and get the next observation
                next_raw_obs, reward, done, _ = env.step(action.item())
                reward = torch.tensor(
                    [[reward]], dtype=torch.float, device=device)

                if done:
                    next_stacked_obs = None
                else:
                    # next_raw_obs = toTensor(next_raw_obs).to(device)
                    # next_raw_obs = next_raw_obs.unsqueeze(0)
                    # next_transformed_obs = transform(next_raw_obs)
                    next_transformed_obs = torch.tensor(
                        next_raw_obs, dtype=torch.float).view(1, 1, 1, 4).to(device)
                    next_stacked_obs = stacking(next_transformed_obs)

                transition = Transition(
                    transformed_obs.to("cpu"),
                    action.to("cpu"),
                    reward.to("cpu"),
                    next_transformed_obs.to("cpu"))
                memory.add(transition)

                stacked_obs = next_stacked_obs

                optimize_model()

                if step % TARGET_NET_UPDATE_FREQ == (TARGET_NET_UPDATE_FREQ - 1):
                    copy_model_state(online_dqn, target_dqn)

                if done:
                    if i_episode % 5 == 4:
                        rollout()

                    if i_episode % 20 == 19:
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
