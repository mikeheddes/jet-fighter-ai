import os
import psutil
import random
from collections import namedtuple
from itertools import count
import gym
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from trainer.env import Environment


# if gpu is to be used
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0
        self.data = []

    def __len__(self):
        return len(self.data)

    def add(self, item):
        if self.wrap_arounds == 0:
            self.data.append(item)
        else:
            self.data[self.write_index] = item

        self.write_index = (self.write_index + 1) % self.capacity
        if self.write_index == 0:
            self.wrap_arounds += 1

    def sample(self, batch_size=1):
        assert len(
            self.data) > batch_size, "Memory contains less data than batch_size"

        return random.sample(self.data, batch_size)


class SumTree:
    """
    Basic implementation of a sum tree
    The parent of two nodes has the value of the sum of its children
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.nodes = [0.] * (capacity * 2 - 1)

    @property
    def sum(self):
        return self.nodes[0]

    def update(self, index, value):
        """
        Update the value of a leave node
        Args
            index: is the external index of the leave node
            value: the new value of the node
        """
        assert index < self.capacity, "Index out of range"

        internal_index = index + self.capacity - 1
        change = value - self.nodes[internal_index]

        self.nodes[internal_index] = value
        self.propagate(internal_index, change)

    def get(self, at_sum):
        """
        Gets the index and value of the node which reached the at_sum value
        Args
            at_sum: the summed value to be reached. Needs to be lower than the total sum of the tree.
        Returns
            index: the index of the leave node
            value: the value of the leave node
        """
        assert at_sum <= self.sum, "Value of at_sum cannot be larger than the sum of the tree"

        internal_index = self.retrieve(0, at_sum)
        index = internal_index - self.capacity + 1

        return index, self.nodes[internal_index]

    def propagate(self, index, change):
        parent = (index - 1) // 2
        self.nodes[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, index, at_sum):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.nodes):
            return index

        if at_sum <= self.nodes[left]:
            return self.retrieve(left, at_sum)
        else:
            return self.retrieve(right, at_sum - self.nodes[left])


class PrioritizedMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.001):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0
        self.data = []

    def __len__(self):
        return len(self.data)

    def add(self, item, error=None):
        if self.wrap_arounds == 0:
            self.data.append(item)
        else:
            self.data[self.write_index] = item

        # Set average error if None
        if error is None:
            error = (self.tree.sum / len(self.data)) ** (1 / self.alpha)

        priority = self.get_priority(error)
        self.tree.update(self.write_index, priority)

        # Update the next write index
        self.write_index = (self.write_index + 1) % self.capacity
        if self.write_index == 0:
            self.wrap_arounds += 1

    def sample(self, batch_size=1):
        transitions = [None] * batch_size
        sample_ids = [None] * batch_size
        is_weights = torch.empty(batch_size, dtype=torch.float)

        for i in range(batch_size):
            at_sum = random.random() * self.tree.sum

            index, priority = self.tree.get(at_sum)
            transitions[i] = self.data[index]
            sample_ids[i] = index + self.wrap_arounds * self.capacity
            probability = priority / self.tree.sum
            is_weights[i] = (len(self.data) * probability) ** -self.beta

        is_weights /= is_weights.max()
        is_weights = is_weights.unsqueeze(1)

        return transitions, sample_ids, is_weights

    def get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def update_priority(self, sample_id, error):
        items_added = self.wrap_arounds * self.capacity + self.write_index - 1

        # Stop if the item to update is no longer in memory
        if sample_id < items_added - self.capacity:
            return

        index = self.sample_id_to_index(sample_id)

        priority = self.get_priority(error)
        self.tree.update(index, priority)

    def sample_id_to_index(self, sample_id):
        return sample_id % self.capacity

    def get_all_priorities(self):
        priorities = self.tree.nodes[-self.tree.capacity:]
        return torch.tensor(priorities, dtype=torch.float)


def get_conv_output_size(size_in, kernel_size, padding=0, stride=1, dilation=1):
    return (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):
    def __init__(self, frame_shape, num_actions):
        super().__init__()
        C, H, W = frame_shape

        self.conv1 = nn.Conv2d(
            in_channels=C, out_channels=16, kernel_size=8, stride=4, padding=2)
        H = get_conv_output_size(H, 8, stride=4, padding=2)
        W = get_conv_output_size(W, 8, stride=4, padding=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        H = get_conv_output_size(H, 4, stride=2, padding=1)
        W = get_conv_output_size(W, 4, stride=2, padding=1)
        self.num_conv2_features = 32 * H * W

        self.fc1 = nn.Linear(self.num_conv2_features, 512)

        # Value layers
        self.vl1 = nn.Linear(512, 128)
        self.vl2 = nn.Linear(128, 1)

        # Action advantage layers
        self.al1 = nn.Linear(512, 128)
        self.al2 = nn.Linear(128, num_actions)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(-1, self.num_conv2_features)
        out = F.relu(self.fc1(out))

        value = F.relu(self.vl1(out))
        value = self.vl2(value)

        advantage = F.relu(self.al1(out))
        advantage = self.al2(advantage)
        mean_advantage = advantage.mean(1, keepdims=True)
        return value + advantage - mean_advantage


BATCH_SIZE = 128
STACKING = 1
GAMMA = 0.997
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5_000_000
LEARNING_RATE = 0.00025
NUM_ACTIONS = 4
MEMORY_SIZE = 500_000
MIN_MEMORY_SIZE = 10_000
TARGET_NET_UPDATE_FREQ = 10_000
NUM_STEPS = 500_000_000
C, H, W = 1, 90, 120

os.makedirs("../checkpoints/", exist_ok=True)
os.makedirs("../runs/", exist_ok=True)


timezone = datetime.timezone.utc
current_date = datetime.datetime.now(timezone)
version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")

writer = SummaryWriter(f'../runs/jet-fighter/{version}/')

# writer.add_hparams({
#     "batch_size": BATCH_SIZE,
#     "gamma": GAMMA,
#     "epsilon_start": EPS_START,
#     "epsilon_end": EPS_END,
#     "epsilon_decay": EPS_DECAY,
#     "learning_rate": LEARNING_RATE,
#     "input_channels": C,
#     "input_height": H,
#     "input_width": W,
#     "number_of_actions": NUM_ACTIONS,
#     "frame_stacking": STACKING,
#     "memory_size": MEMORY_SIZE,
#     "target_network_update_frequency": TARGET_NET_UPDATE_FREQ,
#     "number_of_training_steps": NUM_STEPS,
# }, {}, run_name="/")


memory = PrioritizedMemory(MEMORY_SIZE)


class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(
            [[[[0.2989]], [[0.587]], [[0.114]]]],
            dtype=torch.float)

    def forward(self, x):
        return (x * self.weights).sum(1, keepdims=True)


class Downscale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = torch.full(
            (channels, channels, 2, 2),
            0.25,
            dtype=torch.float)

    def forward(self, x):
        return F.conv2d(x, weight=self.weights, stride=2)


class Transform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gray = Grayscale()
        self.downscaling = Downscale(channels)

    def forward(self, x):
        out = self.gray(x)
        out = self.downscaling(out)
        return out


class Learner:
    def __init__(self, device=None):
        self.online_dqn = DQN((C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.target_dqn = DQN((C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.update_target_model()

        self.optimizer = optim.RMSprop(
            self.online_dqn.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.MSELoss(reduction='none')

    def update_target_model(self):
        state_dict = self.online_dqn.state_dict()
        self.target_dqn.load_state_dict(state_dict)

    def step(self):
        if len(memory) < MIN_MEMORY_SIZE:
            return

        global step
        transitions, sample_ids, is_weights = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = [s is not None for s in batch.next_state]
        non_final_mask = torch.tensor(
            non_final_mask,
            device=device,
            dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(non_final_next_states).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(
            batch.action, device=device, dtype=torch.int64)
        action_batch = action_batch.view(BATCH_SIZE, 1)
        reward_batch = torch.tensor(
            batch.reward, device=device, dtype=torch.float)
        reward_batch = reward_batch.view(BATCH_SIZE, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values = self.online_dqn(state_batch).gather(1, action_batch)

        # Calculate target
        with torch.no_grad():
            non_final_next_actions = self.online_dqn(
                non_final_next_states).argmax(1, keepdims=True)
            next_state_values = torch.zeros((BATCH_SIZE, 1), device=device)
            next_state_values[non_final_mask] = self.target_dqn(
                non_final_next_states).gather(1, non_final_next_actions)
            expected_q_values = reward_batch + (next_state_values * GAMMA)

        errors = torch.abs(q_values - expected_q_values)
        for batch_i in range(BATCH_SIZE):
            memory.update_priority(sample_ids[batch_i], errors[batch_i].item())

        losses = self.loss_fn(q_values, expected_q_values) * is_weights
        loss = losses.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_dqn.parameters(), 1)
        self.optimizer.step()

        if step % 500 == 499:
            print(f"Loss: {loss.item():.4g}, \t at step {step}")
            writer.add_scalar("learner/loss", loss, step)

            state_dict = self.online_dqn.state_dict()
            for tensor_name in state_dict:
                tag = f"learner/{tensor_name}"
                tensor = state_dict[tensor_name]
                writer.add_histogram(tag, tensor, step)

        if step % TARGET_NET_UPDATE_FREQ == TARGET_NET_UPDATE_FREQ - 1:
            self.update_target_model()

        step += 1


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class Actor:
    def __init__(self, model=None, device=None):
        self.device = device
        self.env = Environment()

        if model is None:
            self.model = DQN(
                (STACKING * C, H, W),
                NUM_ACTIONS).to(device)
        else:
            self.model = model

        self.toTensor = T.ToTensor()
        self.transform = Transform(C)

    def policy(self, state):
        global step
        sample = random.random()
        part = 1. - min(step / EPS_DECAY, 1.)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * part
        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).argmax().item()
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    def episode(self):
        state = self.env.reset()
        state = self.toTensor(state).to(self.device)
        state = state.unsqueeze(0)
        state = self.transform(state)

        while True:
            action = self.policy(state)
            next_state, reward, done = self.env.step(action)

            if done:
                next_state = None
            else:
                next_state = self.toTensor(next_state)
                next_state = next_state.unsqueeze(0)
                next_state = self.transform(next_state)

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


class Rollout(Actor):
    def policy(self, state):
        with torch.no_grad():
            q_values = self.model(state)
            self.total_value += q_values.mean().item()
            return q_values.argmax().item()

    def episode(self):
        self.total_value = 0
        self.frames = 0
        for t in super().episode():
            self.frames += 1
            yield t

    @property
    def mean_value(self):
        return self.total_value / self.frames


gpu_mem = torch.cuda.memory_allocated(device) / 1e6
process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / 1e6
print(f"Start script, using {gpu_mem:.2f} MB GPU and {cpu_mem:.2f} MB CPU")

learner = Learner()
actor = Actor(model=learner.online_dqn)
rollout = Rollout(model=learner.online_dqn)

state = actor.env.reset()
state = actor.toTensor(state)
state = state.unsqueeze(0)
state = actor.transform(state)
writer.add_graph(learner.online_dqn, state)

step = 0
start_time = time.time()
for i_episode in count():
    if step >= NUM_STEPS:
        break

    if i_episode % 10 == 9:
        gpu_mem = torch.cuda.memory_allocated(device) / 1e6
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1e6
        writer.add_scalar("host/gpu_memory_usage", gpu_mem, step)
        writer.add_scalar("host/cpu_memory_usage", cpu_mem, step)
        writer.add_scalar("actor/num_episodes", i_episode, step)
        print(
            f"Start episode {i_episode}, using {gpu_mem:.2f} MB GPU and {cpu_mem:.2f} MB CPU")

    for transition in actor.episode():
        memory.add(transition)
        learner.step()

    if i_episode % 5 == 4:
        episode_frames = []
        episode_sum_rewards = 0.0
        for transition in rollout.episode():
            episode_frames.append(transition.state)
            episode_sum_rewards += transition.reward
            memory.add(transition)

        episode_mean_value = rollout.mean_value
        episode_frames = torch.stack(episode_frames, dim=1)
        writer.add_scalar("rollout/episode_sum_rewards",
                          episode_sum_rewards, step)
        writer.add_scalar("rollout/episode_mean_value", episode_mean_value, step)
        writer.add_video("rollout/episode", episode_frames, step, fps=30)
        print("episode summed reward", episode_sum_rewards,
              f"\tmean value: {episode_mean_value:.1f}", "\tat episode", i_episode, f"\tat {(time.time() - start_time):.1f}s")

        writer.add_scalar("memory/length", len(memory), step)
        writer.add_histogram("memory/priorities", memory.get_all_priorities(), step)

    if i_episode % 50 == 49:
        torch.save({
            'steps': step,
            'model_state_dict': learner.online_dqn.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict()
        }, "../checkpoints/training.pt")


actor.env.close()
rollout.env.close()
writer.close()
