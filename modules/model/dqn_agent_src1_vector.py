import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = torch.device("cpu")
print(f"Using device: {device }")


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000
    ):
        """Execute the init routine for review reproduction."""
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self._max_prio_since_last_rescale = 1.0

    def beta_by_frame(self, frame_idx):
        """Execute the beta by frame routine for review reproduction."""
        return min(
            self.beta_end,
            self.beta_start
            + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames,
        )

    def push(self, *args):
        """Execute the push routine for review reproduction."""
        max_prio = self._max_prio_since_last_rescale

        if len(self.buffer) < self.capacity:
            self.buffer.append(args)
        else:
            self.buffer[self.position] = args

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, frame_idx):
        """Execute the sample routine for review reproduction."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
            buffer_len = self.capacity
        else:
            prios = self.priorities[: self.position]
            buffer_len = self.position

        min_prio = prios.min() if prios.size > 0 else 0
        if min_prio == 0 and prios.max() == 0:
            prios = np.ones_like(prios) * 1e-5

        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(prios) / buffer_len
        else:
            probs /= probs_sum

        beta = self.beta_by_frame(frame_idx)

        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]

        weights = (buffer_len * probs[indices]) ** (-beta)
        self._max_prio_since_last_rescale = max(
            1.0, prios.max() if prios.size > 0 else 1.0
        )

        max_weight = (buffer_len * prios.min()) ** (-beta) if prios.min() > 0 else 1.0
        weights /= max_weight
        weights = np.nan_to_num(weights, nan=1e-6)

        return samples, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, priorities):
        """Execute the update priorities routine for review reproduction."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(abs(priority), 1e-6)
            self._max_prio_since_last_rescale = max(
                self._max_prio_since_last_rescale, self.priorities[idx]
            )

    def __len__(self):
        """Execute the len routine for review reproduction."""
        return len(self.buffer)


class BaseDQN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, dropout=0.1):
        """Execute the init routine for review reproduction."""
        super(BaseDQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        """Execute the forward routine for review reproduction."""
        return self.network(x)


class BaseDQNAgent:
    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        lr=0.001,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=50,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100000,
        grad_clip_norm=1.0,
    ):
        """Review package module for dqn agent src1 vector."""
        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.policy_net = BaseDQN(input_dim, hidden_dim).to(self.device)
        self.target_net = BaseDQN(input_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm

        self.memory = PrioritizedReplayBuffer(
            buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=1.0,
            beta_frames=per_beta_frames,
        )

        self.steps_done = 0

    def select_action(self, state, epsilon=None):
        """Execute the select action routine for review reproduction."""
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() > epsilon:
            with torch.no_grad():

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, 1)

    def store_experience(self, state, action, reward, next_state):
        """Execute the store experience routine for review reproduction."""
        self.memory.push(state, action, reward, next_state)

    def train(self):
        """Execute the train routine for review reproduction."""
        if len(self.memory) < self.batch_size:
            return None

        samples, indices, weights = self.memory.sample(self.batch_size, self.steps_done)

        states, actions, rewards, next_states = zip(*samples)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        weights_tensor = weights.to(self.device)

        current_q_values = self.policy_net(states_tensor)
        current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor)
            next_max_q = next_q_values.max(1)[0]
            target_q = rewards_tensor + self.gamma * next_max_q

        td_error = torch.abs(current_q - target_q).detach().cpu().numpy()

        loss = F.mse_loss(current_q, target_q, reduction="none")
        weighted_loss = (weights_tensor * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip_norm
        )
        self.optimizer.step()

        self.memory.update_priorities(indices, td_error)

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon >= self.epsilon_end:
            self.epsilon = max(self.epsilon * 0.996, self.epsilon_end)

        return weighted_loss.item()


if __name__ == "__main__":
    pass
