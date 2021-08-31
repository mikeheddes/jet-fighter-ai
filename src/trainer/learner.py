import torch
import torch.nn as nn
import torch.optim as optim

from trainer.model import DQN
import trainer.config as conf


class Learner:
    def __init__(self, device=None):
        self.device = device

        input_size = (conf.FRAME_STACKING * conf.C, conf.H, conf.W)
        num_outputs = conf.NUM_ACTIONS

        self.online_dqn = DQN(input_size, num_outputs).to(device)
        self.target_dqn = DQN(input_size, num_outputs).to(device)
        self.copy_model_state()

        lr = conf.LEARNING_RATE
        params_to_optimize = self.online_dqn.parameters()
        self.optimizer = optim.RMSprop(params_to_optimize, lr=lr)

        self.loss_fn = nn.MSELoss(reduction='none')

    def step(self, batch, step_idx):
        pred, target = self.get_pred_and_target(batch)

        errors = torch.abs(pred - target)
        for batch_i in range(conf.BATCH_SIZE):
            memory.update_priority(
                sample_ids[batch_i],
                errors[batch_i].item())

        is_weights = is_weights.to(self.device)
        losses = self.loss_fn(pred, target) * is_weights
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step_idx % conf.TARGET_NET_UPDATE_FREQ == (conf.TARGET_NET_UPDATE_FREQ - 1):
            self.copy_model_state()

    def copy_model_state(self):
        state_dict = self.online_dqn.state_dict()
        self.target_dqn.load_state_dict(state_dict)

    def get_pred_and_target(self, batch):
        non_final_mask = [s is not None for s in batch.next_state]
        non_final_mask = torch.tensor(
            non_final_mask, dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Current prediction
        q_values = self.online_dqn(state_batch).gather(1, action_batch)

        # Calculate target
        with torch.no_grad():
            non_final_next_actions = self.online_dqn(
                non_final_next_states).argmax(1, keepdims=True)
            next_state_values = torch.zeros((conf.BATCH_SIZE, 1), device=self.device)
            next_state_values[non_final_mask] = self.target_dqn(
                non_final_next_states).gather(1, non_final_next_actions)
            discount = conf.GAMMA ** conf.MULTI_STEP
            expected_q_values = reward_batch + (next_state_values * discount)

        return q_values, expected_q_values
