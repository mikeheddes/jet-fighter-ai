import torch
import torch.nn as nn
import torch.optim as optim
import ray
import time

from .types import Transition
from .model import DQN
from .globals import BATCH_SIZE, NUM_STEPS, STACKING, C, H, W, NUM_ACTIONS, LEARNING_RATE, GAMMA, TARGET_NET_UPDATE_FREQ, MIN_MEMORY_SIZE


@ray.remote
class Learner:
    def __init__(self, device=None):
        self.device = device
        self.online_dqn = DQN((1, C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.target_dqn = DQN((1, C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.update_target_model()

        parameters = self.online_dqn.parameters()
        self.optimizer = optim.RMSprop(parameters, lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss(reduction='none')

        self.commons = ray.get_actor("commons")
        self.internal_step_count = ray.get(self.commons.get_step.remote())
        self.update_step_countdown = 0

        self.cached_is_memory_sufficiently_full = False
        self.fetch_memory_size_countdown = 0

        self.update_global_model()

    def update_target_model(self):
        state_dict = self.online_dqn.state_dict()
        self.target_dqn.load_state_dict(state_dict)

    def set_step_count(self, count):
        self.internal_step_count = count
        self.update_step_countdown -= 1

        if self.update_step_countdown <= 0:
            self.update_step_countdown = 50
            self.commons.set_step.remote(self.internal_step_count)

    def get_step_count(self):
        return self.internal_step_count

    def is_memory_sufficiently_full(self, memory):
        if self.cached_is_memory_sufficiently_full:
            return True

        if self.fetch_memory_size_countdown <= 0:
            self.fetch_memory_size_countdown = 100
            self.cached_is_memory_sufficiently_full = ray.get(
                memory.__len__.remote()) >= MIN_MEMORY_SIZE

        self.fetch_memory_size_countdown -= 1
        return self.cached_is_memory_sufficiently_full

    def update_global_model(self):
        self.commons.set_model_state_dict.remote(
            self.online_dqn.state_dict())

    def step(self, transitions, sample_ids, is_weights, memory):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = [s is not None for s in batch.next_state]
        non_final_mask = torch.tensor(
            non_final_mask,
            device=self.device,
            dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(
            non_final_next_states).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.int64)
        action_batch = action_batch.view(BATCH_SIZE, 1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float)
        reward_batch = reward_batch.view(BATCH_SIZE, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values = self.online_dqn(state_batch).gather(1, action_batch)

        # Calculate target
        with torch.no_grad():
            non_final_next_actions = self.online_dqn(
                non_final_next_states).argmax(1, keepdims=True)
            next_state_values = torch.zeros(
                (BATCH_SIZE, 1), device=self.device)
            next_state_values[non_final_mask] = self.target_dqn(
                non_final_next_states).gather(1, non_final_next_actions)
            expected_q_values = reward_batch + (next_state_values * GAMMA)

        errors = torch.abs(q_values - expected_q_values).view(BATCH_SIZE)
        memory.update_priorities.remote(sample_ids, errors.tolist())

        is_weights = is_weights.to(self.device)
        losses = self.loss_fn(q_values, expected_q_values) * is_weights
        loss = losses.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_dqn.parameters(), 1.0)
        self.optimizer.step()

        step = self.get_step_count()
        if step % 500 == 499:
            self.commons.write_scalar.remote("learner/loss", loss)

            state_dict = self.online_dqn.state_dict()
            for tensor_name in state_dict:
                tag = f"learner/{tensor_name}"
                tensor = state_dict[tensor_name]
                self.commons.write_histogram.remote(tag, tensor)

        if step % TARGET_NET_UPDATE_FREQ == TARGET_NET_UPDATE_FREQ - 1:
            self.update_target_model()
            self.update_global_model()

        if step % 5000 == 4999:
            torch.save({
                'steps': step,
                'model_state_dict': self.online_dqn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, "./checkpoints/training.pt")

        self.set_step_count(step + 1)

    def train(self):
        memory = ray.get_actor("memory")

        while not self.is_memory_sufficiently_full(memory):
            time.sleep(0.01)

        fetch_batch = memory.sample.remote(BATCH_SIZE)
        for _ in range(NUM_STEPS):
            transitions, sample_ids, is_weights = ray.get(fetch_batch)
            fetch_batch = memory.sample.remote(BATCH_SIZE)
            self.step(transitions, sample_ids, is_weights, memory)
