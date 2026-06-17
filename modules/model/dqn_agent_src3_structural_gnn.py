import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch


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


class DQN(nn.Module):
    def __init__(self, history_dim=2, gru_hidden_dim=64, gnn_hidden=128, dropout=0.1):
        """Execute the init routine for review reproduction."""
        super(DQN, self).__init__()

        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.gcn1 = GCNConv(gru_hidden_dim, gnn_hidden)
        self.gcn2 = GCNConv(gnn_hidden, gnn_hidden)

        self.q_mlp = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden // 2, 2),
        )

        self.gru_hidden_dim = gru_hidden_dim

    def forward(self, data):
        """Execute the forward routine for review reproduction."""

        if isinstance(data, dict):
            x, edge_index = data["x"], data["edge_index"]
        else:
            x, edge_index = data.x, data.edge_index

        batch_size = 1
        is_batch = hasattr(data, "batch") and data.batch is not None

        _, seq_len, hist_dim = x.size()

        x_flat = x.view(-1, seq_len, hist_dim)  # [batch_nodes, seq_len, hist_dim]

        _, h_n = self.gru(x_flat)  # h_n: [1, batch_nodes, gru_hidden_dim]

        node_embeddings = h_n.squeeze(0)  # [batch_nodes, gru_hidden_dim]

        x = self.gcn1(node_embeddings, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        if is_batch:
            batch = data.batch
            batch_size = batch.max().item() + 1

            agent_indices = []
            for b in range(batch_size):

                indices = (batch == b).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    agent_indices.append(indices[0].item())
                else:

                    agent_indices.append(0)

            agent_indices = torch.tensor(agent_indices, device=x.device)

            agent_features = x[agent_indices]
        else:

            agent_features = x[0].unsqueeze(0)

        q_values = self.q_mlp(agent_features)  # [batch_size, 2]

        return q_values


class DQNAgent:
    def __init__(
        self,
        history_dim=2,
        gru_hidden_dim=16,
        gnn_hidden=16,
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
        """Review package module for dqn agent src3 structural gnn."""
        self.device = device
        self.history_dim = history_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gnn_hidden = gnn_hidden

        self.policy_net = DQN(history_dim, gru_hidden_dim, gnn_hidden).to(self.device)
        self.target_net = DQN(history_dim, gru_hidden_dim, gnn_hidden).to(self.device)
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
        self.per_beta_frames = per_beta_frames

        self.steps_done = 0

    def select_action(self, state, epsilon=None):
        """Execute the select action routine for review reproduction."""

        if epsilon is None:
            epsilon = self.epsilon

        if random.random() > epsilon:
            with torch.no_grad():

                state_data = state.to(self.device)

                q_values = self.policy_net(state_data)

                return q_values.argmax().item()
        else:
            return random.randint(0, 1)

    def store_experience(self, state, action, reward, next_state):
        """Execute the store experience routine for review reproduction."""

        state_cpu = state.cpu()
        next_state_cpu = next_state.cpu()

        self.memory.push(state_cpu, action, reward, next_state_cpu)

    def batch_states(self, states):
        """Execute the batch states routine for review reproduction."""

        batch = Batch.from_data_list(states).to(self.device)

        return batch

    def train(self):
        """Execute the train routine for review reproduction."""

        if len(self.memory) < self.batch_size:
            return None

        samples, indices, weights = self.memory.sample(self.batch_size, self.steps_done)

        states, actions, rewards, next_states = zip(*samples)

        batch_states = self.batch_states(states)
        batch_next_states = self.batch_states(next_states)

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        weights_tensor = weights.to(self.device)

        current_q_values = self.policy_net(batch_states)
        current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(batch_next_states)
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
