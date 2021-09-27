import torch
import torch.nn as nn
import torch.optim as optim

from .utils import do_every
from .settings import BATCH_SIZE, STACKING, C, H, W, NUM_ACTIONS, LEARNING_RATE, GAMMA, TARGET_NET_UPDATE_FREQ


class Learner:
    def __init__(self, dqn_cls, step, device=None):
        self.step_count = step
        self.device = device

        self.online_dqn = dqn_cls((C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.target_dqn = dqn_cls((C * STACKING, H, W), NUM_ACTIONS).to(device)
        self.update_target_model()

        parameters = self.online_dqn.parameters()
        self.optimizer = optim.RMSprop(parameters, lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss(reduction='none')

        self.last_loss = 0.0

    def update_target_model(self):
        state_dict = self.online_dqn.state_dict()
        self.target_dqn.load_state_dict(state_dict)

    def get_prediction_and_expectation(self, batch):
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

        return q_values, expected_q_values

    def step(self, batch, is_weights):
        pred, expected = self.get_prediction_and_expectation(batch)

        with torch.no_grad():
            errors = torch.abs(pred - expected)

        is_weights = is_weights.to(self.device)
        losses = self.loss_fn(pred, expected) * is_weights
        loss = losses.mean()
        self.last_loss = loss.item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_dqn.parameters(), 1.0)
        self.optimizer.step()

        if do_every(TARGET_NET_UPDATE_FREQ, self.step_count.value):
            self.update_target_model()

        with self.step_count.get_lock():
            self.step_count.value += 1

        return errors


    def metrics(self, step):
        yield ("scalar", "learner/loss", self.last_loss, step.value)

        state_dict = self.online_dqn.state_dict()
        for tensor_name in state_dict:
            tag = f"learner/{tensor_name}"
            tensor = state_dict[tensor_name]
            yield ("histogram", tag, tensor, step.value)
