import torch
import torch.nn as nn
import torch.optim as optim
import ray

from .types import Transition
from .model import DQN
from .globals import BATCH_SIZE, STACKING, C, H, W, NUM_ACTIONS, LEARNING_RATE, GAMMA, TARGET_NET_UPDATE_FREQ

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

    def update_target_model(self):
        state_dict = self.online_dqn.state_dict()
        self.target_dqn.load_state_dict(state_dict)

    def step(self, memory):
        # TODO: Cache call and invalidate every n-calls 
        # until memory is long enough,
        # then cache value forever.
        if ray.get(memory.__len__.remote()) < 1000:
            return

        transitions, sample_ids, is_weights = ray.get(memory.sample.remote(BATCH_SIZE))
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

        # TODO use internal value counter for step
        # update the commons counter every n-calls
        step = ray.get(self.commons.get_step.remote())
        if step % 500 == 499:
            self.commons.write_scalar.remote("learner/loss", loss)

            state_dict = self.online_dqn.state_dict()
            for tensor_name in state_dict:
                tag = f"learner/{tensor_name}"
                tensor = state_dict[tensor_name]
                self.commons.write_histogram.remote(tag, tensor)

        if step % TARGET_NET_UPDATE_FREQ == TARGET_NET_UPDATE_FREQ - 1:
            self.update_target_model()

        # TODO use internal value counter for step
        # update the commons counter every n-calls
        self.commons.set_step.remote(step + 1)
