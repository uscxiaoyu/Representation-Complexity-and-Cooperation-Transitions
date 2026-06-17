import pickle
import random
import time
import os
from collections import Counter, deque

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from modules.model.dqn_agent_src1_vector import BaseDQNAgent as DQNAgentSimple
from modules.model.dqn_agent_src2_relational import DQNAgent as DQNAgentMedium
from modules.model.dqn_agent_src3_structural_gnn import DQNAgent as DQNAgentComplex
from modules.model.checkpoint_io import save_checkpoint
from modules.paths import PROJECT_PATHS


def set_seed(random_seed):
    """Execute the set seed routine for review reproduction."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


class CooperationABM:
    def __init__(
        self,
        n=1000,
        payoffs=(1.0, 0.0, 1.5, 0.1),
        alpha=2.0,
        c=0.05,
        p_triadic=0.5,
        omega=0.5,
        epsilon=0.005,
        rho=0.1,
        mu=0.1,
        trust_strategy="Open",
        drl_agent_ratio=0.1,
        max_memory=5,
        pre_train_epochs=3000,
        train_epochs=3000,
        test_epochs=3000,
        dqn_agent_params=None,
        environment_complexity=1,
        dynamic_payoff=False,
        perturbation_interval=500,
        trust_disturbance_rate=0.1,
        input_complexity="simple",
        save_checkpoints=False,
        checkpoint_dir="model_checkpoints",
        checkpoint_intervals=None,
        checkpoint_frequency=None,
        experiment_id=None,
    ):
        """Review package module for adaptive network model."""
        self.n = n
        self.environment_complexity = environment_complexity
        self.dynamic_payoff = dynamic_payoff
        self.perturbation_interval = perturbation_interval
        self.trust_disturbance_rate = trust_disturbance_rate
        self.input_complexity = input_complexity
        self.max_memory = max_memory

        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_intervals = checkpoint_intervals or []
        self.checkpoint_frequency = checkpoint_frequency
        self.experiment_id = experiment_id

        self.r_original = payoffs[0]  # rewarding payoff
        self.s_original = payoffs[1]  # sucker's payoff
        self.t_original = payoffs[2]  # temptation payoff
        self.p_original = payoffs[3]  # punishment payoff

        self.r = self.r_original
        self.s = self.s_original
        self.t = self.t_original
        self.p = self.p_original

        self.payoff_matrix = {
            (1, 1): (self.r, self.r),
            (1, 0): (self.s, self.t),
            (0, 1): (self.t, self.s),
            (0, 0): (self.p, self.p),
        }

        self.c = c  # cost of maintaining a link
        self.alpha = alpha  # cost of maintaining a link with k neighbors

        self.p_triadic = p_triadic  # triadic closure probability
        self.omega = omega  # trust update rule weight
        self.epsilon = epsilon  # trust random-reset error rate

        self.mu = mu  # partner update rate
        self.rho = rho  # strategy update rate

        self.trust_strategy = trust_strategy  # trust strategy
        self.drl_agent_ratio = drl_agent_ratio  # ratio of DRL agents

        self.pre_train_epochs = pre_train_epochs
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))

        if dqn_agent_params is None:
            dqn_agent_params = {}

        if self.input_complexity == "complex":
            self.drl_agent = DQNAgentComplex(**dqn_agent_params)
        elif self.input_complexity == "medium":
            self.drl_agent = DQNAgentMedium(**dqn_agent_params)
        else:
            self.drl_agent = DQNAgentSimple(**dqn_agent_params)

        num_drl_agents = int(self.n * self.drl_agent_ratio)
        self.drl_nodes = np.random.choice(
            self.G.nodes(), num_drl_agents, replace=False
        ).tolist()
        self.imitation_nodes = [i for i in self.G.nodes() if i not in self.drl_nodes]
        for i in self.G.nodes():
            if i in self.drl_nodes:
                self.G.nodes[i]["strategy"] = "drl"
            else:
                self.G.nodes[i]["strategy"] = "imitation"

            self.G.nodes[i]["state"] = None

        self.new_neighbors = {i: [] for i in self.G.nodes()}
        self.current_drl_states = {}

        for i in self.G.nodes():
            self.G.nodes[i]["current_payoff"] = 0
            self.G.nodes[i]["current_strategy"] = np.random.choice([0, 1])
            self.G.nodes[i]["previous_strategy"] = self.G.nodes[i]["current_strategy"]
            self.G.nodes[i]["trust"] = np.random.uniform()

            if self.input_complexity == "complex":
                self.G.nodes[i]["payoff_history"] = deque(maxlen=self.max_memory)
                self.G.nodes[i]["strategy_history"] = deque(maxlen=self.max_memory)

    def update_payoff_matrix(self):
        """Execute the update payoff matrix routine for review reproduction."""
        self.payoff_matrix = {
            (1, 1): (self.r, self.r),
            (1, 0): (self.s, self.t),
            (0, 1): (self.t, self.s),
            (0, 0): (self.p, self.p),
        }

    def apply_perturbation(self, current_epoch):
        """Execute the apply perturbation routine for review reproduction."""
        if self.environment_complexity == 3 and self.dynamic_payoff:
            if current_epoch % self.perturbation_interval == 0 and current_epoch > 0:

                self.t, self.s = self.s, self.t
                self.update_payoff_matrix()
                print(
                    f"Periodic perturbation applied at round {current_epoch } round: T={self .t :.1f}, S={self .s :.1f}"
                )

    def calculate_payoffs(self):
        """Execute the calculate payoffs routine for review reproduction."""
        payoffs = np.zeros(self.n)

        for i, j in self.G.edges():
            strategy_i, strategy_j = (
                self.G.nodes[i]["current_strategy"],
                self.G.nodes[j]["current_strategy"],
            )
            payoff_i, payoff_j = self.payoff_matrix[(strategy_i, strategy_j)]
            payoffs[i] += payoff_i
            payoffs[j] += payoff_j

        for i in self.G.nodes():
            degree = self.G.degree(i)
            payoffs[i] = (
                payoffs[i] - self.c * (degree ** self.alpha) if degree > 0 else 0
            )
            self.G.nodes[i]["current_payoff"] = payoffs[i]

        return payoffs

    def store_previous_strategy(self):
        """Execute the store previous strategy routine for review reproduction."""
        for i in self.G.nodes():
            self.G.nodes[i]["previous_strategy"] = self.G.nodes[i]["current_strategy"]

    def store_history_data(self):
        """Execute the store history data routine for review reproduction."""
        if self.input_complexity == "complex":
            for i in self.G.nodes():
                self.G.nodes[i]["payoff_history"].append(
                    self.G.nodes[i]["current_payoff"]
                )
                self.G.nodes[i]["strategy_history"].append(
                    self.G.nodes[i]["current_strategy"]
                )

    def _get_model_config(self):
        """Execute the get model config routine for review reproduction."""
        if self.input_complexity == "simple":

            return {
                "input_complexity": "simple",
                "input_dim": self.drl_agent.input_dim,
                "hidden_dim": self.drl_agent.hidden_dim,
            }
        elif self.input_complexity == "medium":

            return {
                "input_complexity": "medium",
                "input_dim": self.drl_agent.input_dim,
                "hidden_dim": self.drl_agent.hidden_dim,
            }
        elif self.input_complexity == "complex":

            return {
                "input_complexity": "complex",
                "history_dim": self.drl_agent.history_dim,
                "gru_hidden_dim": self.drl_agent.gru_hidden_dim,
                "gnn_hidden": self.drl_agent.gnn_hidden,
            }
        else:
            return {"input_complexity": self.input_complexity}

    def _save_checkpoint(self, round_num, phase="train"):
        """Execute the save checkpoint routine for review reproduction."""
        if not self.save_checkpoints:
            return

        try:

            if self.experiment_id is None:
                exp_id = f"ec{self .environment_complexity }_ic{self .input_complexity }_ratio{self .drl_agent_ratio }"
            else:
                exp_id = self.experiment_id

            os.makedirs(self.checkpoint_dir, exist_ok=True)

            if phase.endswith("_final"):
                checkpoint_name = f"{exp_id }_{phase }.pt"
            else:
                checkpoint_name = f"{exp_id }_{phase }_round{round_num }.pt"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            additional_info = {
                "environment_complexity": self.environment_complexity,
                "drl_agent_ratio": self.drl_agent_ratio,
                "phase": phase,
                "n_agents": self.n,
            }

            save_checkpoint(
                self.drl_agent, checkpoint_path, round_num, exp_id, additional_info
            )

        except Exception as e:
            print(f"Warning: Failed to save checkpoint (round {round_num }): {e }")

    def prepare_state(self, i, t=None):
        """Execute the prepare state routine for review reproduction."""
        if self.input_complexity == "simple":
            return self.prepare_simple_state(i)
        elif self.input_complexity == "medium":
            return self.prepare_medium_state(i)
        elif self.input_complexity == "complex":
            return self.prepare_complex_state(i, t)
        else:
            raise ValueError(f"Unknown input_complexity: {self .input_complexity }")

    def prepare_simple_state(self, i):
        """Execute the prepare simple state routine for review reproduction."""
        neighbors = list(self.G.neighbors(i))

        own_previous_strategy = float(self.G.nodes[i]["previous_strategy"])

        if neighbors:
            neighbor_cooperation_rate = np.mean(
                [self.G.nodes[j]["previous_strategy"] for j in neighbors]
            )
        else:
            neighbor_cooperation_rate = 0.0

        normalized_neighbor_count = len(neighbors) / (self.n - 1)

        state = [
            own_previous_strategy,
            neighbor_cooperation_rate,
            normalized_neighbor_count,
        ]
        self.G.nodes[i]["state"] = state
        return state

    def prepare_medium_state(self, i):
        """Execute the prepare medium state routine for review reproduction."""
        neighbors = list(self.G.neighbors(i))

        own_previous_strategy = float(self.G.nodes[i]["previous_strategy"])

        if neighbors:
            neighbor_cooperation_rate = np.mean(
                [self.G.nodes[j]["previous_strategy"] for j in neighbors]
            )
            neighbor_avg_payoff = np.mean(
                [self.G.nodes[j]["current_payoff"] for j in neighbors]
            )
            neighbor_avg_trust = np.mean([self.G.nodes[j]["trust"] for j in neighbors])
        else:
            neighbor_cooperation_rate = 0.0
            neighbor_avg_payoff = 0.0
            neighbor_avg_trust = 0.5

        normalized_neighbor_count = len(neighbors) / (self.n - 1)

        own_payoff_normalized = np.tanh(self.G.nodes[i]["current_payoff"])
        neighbor_payoff_normalized = np.tanh(neighbor_avg_payoff)

        state_list = [
            own_previous_strategy,
            neighbor_cooperation_rate,
            normalized_neighbor_count,
            own_payoff_normalized,
            neighbor_payoff_normalized,
            self.G.nodes[i]["trust"],
            neighbor_avg_trust,
        ]

        state_tensor = torch.FloatTensor(state_list)
        self.G.nodes[i]["state"] = state_tensor
        return state_tensor

    def prepare_complex_state(self, i, t):
        """Execute the prepare complex state routine for review reproduction."""
        neighbors = list(self.G.neighbors(i))

        own_payoff = list(self.G.nodes[i]["payoff_history"])
        own_strategy = list(self.G.nodes[i]["strategy_history"])

        if len(own_payoff) < self.max_memory:
            padding_length = self.max_memory - len(own_payoff)
            own_payoff = [0] * padding_length + own_payoff
            own_strategy = [0] * padding_length + own_strategy

        node_features = [torch.FloatTensor([own_payoff, own_strategy]).T]

        if not neighbors:
            x = node_features[0].unsqueeze(0)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            self.G.nodes[i]["state"] = Data(x=x, edge_index=edge_index)
            return self.G.nodes[i]["state"]

        for j in neighbors:

            if (
                "payoff_history" not in self.G.nodes[j]
                or "strategy_history" not in self.G.nodes[j]
            ):

                self.G.nodes[j]["payoff_history"] = deque(maxlen=self.max_memory)
                self.G.nodes[j]["strategy_history"] = deque(maxlen=self.max_memory)

            if t is not None and "build_time" in self.G.edges[i, j]:
                history_length = min(
                    self.max_memory, t - self.G.edges[i, j]["build_time"]
                )
            else:
                history_length = self.max_memory

            neighbor_payoff = list(self.G.nodes[j]["payoff_history"])[-history_length:]
            neighbor_strategy = list(self.G.nodes[j]["strategy_history"])[
                -history_length:
            ]

            if len(neighbor_payoff) < self.max_memory:
                padding_length = self.max_memory - len(neighbor_payoff)
                neighbor_payoff = [0] * padding_length + neighbor_payoff
                neighbor_strategy = [0] * padding_length + neighbor_strategy

            node_features.append(
                torch.FloatTensor([neighbor_payoff, neighbor_strategy]).T
            )

        subgraph_nodes = [i] + neighbors
        node_map = {node_id: idx for idx, node_id in enumerate(subgraph_nodes)}

        edge_index = []
        for u, v in self.G.subgraph(subgraph_nodes).edges():
            new_u, new_v = node_map[u], node_map[v]
            edge_index.extend([[new_u, new_v], [new_v, new_u]])

        x = torch.stack(node_features)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        self.G.nodes[i]["state"] = Data(x=x, edge_index=edge_index)
        return self.G.nodes[i]["state"]

    def update_strategy_by_imitation(self, i):
        """Execute the update strategy by imitation routine for review reproduction."""

        if random.random() < self.rho:
            neighbors = list(self.G.neighbors(i))
            if neighbors:

                best_neighbor = max(
                    neighbors, key=lambda x: self.G.nodes[x]["current_payoff"]
                )
                if (
                    self.G.nodes[best_neighbor]["current_payoff"]
                    > self.G.nodes[i]["current_payoff"]
                ):
                    self.G.nodes[i]["current_strategy"] = self.G.nodes[best_neighbor][
                        "current_strategy"
                    ]

    def update_strategy_by_drl(self, i, is_testing=False):
        """Execute the update strategy by drl routine for review reproduction."""
        with torch.no_grad():

            epsilon = 0.0 if is_testing else None
            self.G.nodes[i]["current_strategy"] = self.drl_agent.select_action(
                self.G.nodes[i]["state"], epsilon=epsilon
            )

    def update_neighbors(self, i, t):
        """Execute the update neighbors routine for review reproduction."""
        if random.random() < self.mu:
            if random.random() < 0.5:
                if not self.try_cut_a_tie(i):
                    self.try_add_a_tie(i, t)
            else:
                if not self.try_add_a_tie(i, t):
                    self.try_cut_a_tie(i)

    def try_cut_a_tie(self, i):
        """Execute the try cut a tie routine for review reproduction."""
        num_neighbors = self.G.degree(i)
        if num_neighbors == 0:
            return False

        j = random.choice(list(self.G.neighbors(i)))

        marginal_benefit = self.payoff_matrix[
            (self.G.nodes[i]["current_strategy"], self.G.nodes[j]["current_strategy"])
        ][0]

        should_cut = (
            self.c * (num_neighbors ** self.alpha - (num_neighbors - 1) ** self.alpha)
            > marginal_benefit
        )
        cut_probability = 1 - self.mu if should_cut else self.mu
        if random.random() < cut_probability:
            self.G.remove_edge(i, j)
            return True

        return False

    def try_add_a_tie(self, i, t):
        """Execute the try add a tie routine for review reproduction."""

        if random.random() < self.p_triadic:
            neighbors = list(self.G.neighbors(i))
            candidates = []
            for j in neighbors:
                candidates.extend(set(self.G.neighbors(j)) - {i} - set(neighbors))

            if candidates:

                freq = Counter(candidates)
                nodes, weights = zip(*freq.items())
                j = np.random.choice(nodes, p=np.array(weights) / sum(weights))

                if self.evaluate_build_link(i, j, is_neighbor_neighbor=True):
                    self.G.add_edge(i, j, build_time=t)
                    return True
            return False

        j = random.choice(list(self.G.nodes))
        if self.evaluate_build_link(i, j, is_neighbor_neighbor=False):
            self.G.add_edge(i, j, build_time=t)
            return True
        return False

    def calculate_expected_benefit(self, strategy, trust_value):
        """Execute the calculate expected benefit routine for review reproduction."""
        if strategy == 1:
            return trust_value * self.r + (1 - trust_value) * self.s

        return trust_value * self.t + (1 - trust_value) * self.p

    def evaluate_build_link(self, i, j, is_neighbor_neighbor=True):
        """Execute the evaluate build link routine for review reproduction."""
        num_neighbors_i, num_neighbors_j = self.G.degree(i), self.G.degree(j)
        strategy_i, strategy_j = (
            self.G.nodes[i]["current_strategy"],
            self.G.nodes[j]["current_strategy"],
        )

        marginal_cost_i = self.c * (
            (num_neighbors_i + 1) ** self.alpha - num_neighbors_i ** self.alpha
        )
        marginal_cost_j = self.c * (
            (num_neighbors_j + 1) ** self.alpha - num_neighbors_j ** self.alpha
        )

        if is_neighbor_neighbor:

            marginal_benefit_i, marginal_benefit_j = self.payoff_matrix[
                (strategy_i, strategy_j)
            ]
        else:

            marginal_benefit_i = self.calculate_expected_benefit(
                strategy_i, self.G.nodes[i]["trust"]
            )
            marginal_benefit_j = self.calculate_expected_benefit(
                strategy_j, self.G.nodes[j]["trust"]
            )

        return (
            marginal_cost_i < marginal_benefit_i
            and marginal_cost_j < marginal_benefit_j
        )

    def update_trust(self, i):
        """Execute the update trust routine for review reproduction."""

        if self.G.degree(i) > 0:
            neighbors = (
                list(self.G.neighbors(i))
                if self.trust_strategy == "Open"
                else self.new_neighbors[i]
            )
            if neighbors:
                ratio = np.mean(
                    [self.G.nodes[j]["current_strategy"] for j in neighbors]
                )
                self.G.nodes[i]["trust"] = (
                    self.omega * self.G.nodes[i]["trust"] + (1 - self.omega) * ratio
                )

        if random.random() < self.epsilon:
            self.G.nodes[i]["trust"] = random.random()

        if (
            self.environment_complexity == 3
            and random.random() < self.trust_disturbance_rate
        ):

            disturbance = np.random.normal(0, 0.1)
            self.G.nodes[i]["trust"] = np.clip(
                self.G.nodes[i]["trust"] + disturbance, 0.0, 1.0
            )

    def pre_train(self):
        """Execute the pre train routine for review reproduction."""

        pre_train_payoffs_history = []
        pre_train_cooperation_rate_history = []
        pre_train_avg_degree_history = []
        pre_train_avg_trust_history = []
        pre_train_clustering_history = []
        pre_train_assortativity_history = []
        pre_train_largest_component_history = []

        t1 = time.perf_counter()
        for t in range(1, self.pre_train_epochs + 1):

            self.apply_perturbation(t)

            payoffs = self.calculate_payoffs()

            self.store_previous_strategy()

            self.new_neighbors = {i: [] for i in self.G.nodes()}
            for i in self.G.nodes():
                self.update_trust(i)
                self.update_neighbors(i, t)

            for i in self.G.nodes():
                self.update_strategy_by_imitation(i)

            self.store_history_data()

            avg_payoffs = np.mean(payoffs)
            pre_train_payoffs_history.append(avg_payoffs)
            pre_train_cooperation_rate_history.append(
                np.mean([self.G.nodes[i]["current_strategy"] for i in self.G.nodes()])
            )
            pre_train_avg_degree_history.append(
                np.mean([d for _, d in self.G.degree()])
            )
            pre_train_avg_trust_history.append(
                np.mean([self.G.nodes[i]["trust"] for i in self.G.nodes()])
            )

            clustering = nx.average_clustering(self.G)
            pre_train_clustering_history.append(clustering)

            for node in self.G.nodes():
                self.G.nodes[node]["node_type"] = (
                    "drl" if node in self.drl_nodes else "imitation"
                )
            assortativity = nx.attribute_assortativity_coefficient(self.G, "node_type")
            pre_train_assortativity_history.append(assortativity)

            if nx.is_connected(self.G):
                largest_component_size = self.G.number_of_nodes()
            else:
                largest_component = max(nx.connected_components(self.G), key=len)
                largest_component_size = len(largest_component)
            pre_train_largest_component_history.append(largest_component_size)

            if t % 500 == 0:
                print(
                    f"Pre-training progress: {t }/{self .pre_train_epochs }, elapsed: {time .perf_counter ()-t1 :.2f}s"
                )
                t1 = time.perf_counter()

        for i in self.drl_nodes:
            self.current_drl_states[i] = self.prepare_state(i, t)

        return (
            pre_train_payoffs_history,
            pre_train_cooperation_rate_history,
            pre_train_avg_degree_history,
            pre_train_avg_trust_history,
            pre_train_clustering_history,
            pre_train_assortativity_history,
            pre_train_largest_component_history,
        )

    def train(self):
        """Execute the train routine for review reproduction."""

        payoffs_history = []
        cooperation_rate_history = []
        avg_degree_history = []
        avg_trust_history = []
        clustering_history = []
        assortativity_history = []
        largest_component_history = []

        drl_payoffs_history = []
        drl_cooperation_rate_history = []
        drl_avg_degree_history = []
        drl_avg_trust_history = []

        imitation_payoffs_history = []
        imitation_cooperation_rate_history = []
        imitation_avg_degree_history = []
        imitation_avg_trust_history = []

        t1 = time.perf_counter()
        for t in range(1, self.train_epochs + 1):

            self.apply_perturbation(t + self.pre_train_epochs)

            # behavior dynamics
            payoffs = self.calculate_payoffs()

            self.store_previous_strategy()

            self.new_neighbors = {i: [] for i in self.G.nodes()}
            for i in self.G.nodes():
                self.update_trust(i)
                self.update_neighbors(i, t + self.pre_train_epochs)

            for i in self.drl_nodes:
                self.update_strategy_by_drl(i)

            for i in self.imitation_nodes:
                self.update_strategy_by_imitation(i)

            self.store_history_data()

            avg_payoffs = np.mean(payoffs)
            payoffs_history.append(avg_payoffs)
            cooperation_rate_history.append(
                np.mean([self.G.nodes[i]["current_strategy"] for i in self.G.nodes()])
            )
            avg_degree_history.append(np.mean([d for _, d in self.G.degree()]))
            avg_trust_history.append(
                np.mean([self.G.nodes[i]["trust"] for i in self.G.nodes()])
            )

            clustering = nx.average_clustering(self.G)
            clustering_history.append(clustering)

            for node in self.G.nodes():
                self.G.nodes[node]["node_type"] = (
                    "drl" if node in self.drl_nodes else "imitation"
                )
            assortativity = nx.attribute_assortativity_coefficient(self.G, "node_type")
            assortativity_history.append(assortativity)

            if nx.is_connected(self.G):
                largest_component_size = self.G.number_of_nodes()
            else:
                largest_component = max(nx.connected_components(self.G), key=len)
                largest_component_size = len(largest_component)
            largest_component_history.append(largest_component_size)

            if len(self.drl_nodes) > 0:
                drl_payoffs = [payoffs[i] for i in self.drl_nodes]
                drl_strategies = [
                    self.G.nodes[i]["current_strategy"] for i in self.drl_nodes
                ]
                drl_degrees = [d for i, d in self.G.degree() if i in self.drl_nodes]
                drl_trusts = [self.G.nodes[i]["trust"] for i in self.drl_nodes]

                drl_payoffs_history.append(np.mean(drl_payoffs) if drl_payoffs else 0)
                drl_cooperation_rate_history.append(
                    np.mean(drl_strategies) if drl_strategies else 0
                )
                drl_avg_degree_history.append(
                    np.mean(drl_degrees) if drl_degrees else 0
                )
                drl_avg_trust_history.append(np.mean(drl_trusts) if drl_trusts else 0)
            else:
                drl_payoffs_history.append(0)
                drl_cooperation_rate_history.append(0)
                drl_avg_degree_history.append(0)
                drl_avg_trust_history.append(0)

            if len(self.imitation_nodes) > 0:
                imitation_payoffs = [payoffs[i] for i in self.imitation_nodes]
                imitation_strategies = [
                    self.G.nodes[i]["current_strategy"] for i in self.imitation_nodes
                ]
                imitation_degrees = [
                    d for i, d in self.G.degree() if i in self.imitation_nodes
                ]
                imitation_trusts = [
                    self.G.nodes[i]["trust"] for i in self.imitation_nodes
                ]

                imitation_payoffs_history.append(
                    np.mean(imitation_payoffs) if imitation_payoffs else 0
                )
                imitation_cooperation_rate_history.append(
                    np.mean(imitation_strategies) if imitation_strategies else 0
                )
                imitation_avg_degree_history.append(
                    np.mean(imitation_degrees) if imitation_degrees else 0
                )
                imitation_avg_trust_history.append(
                    np.mean(imitation_trusts) if imitation_trusts else 0
                )
            else:
                imitation_payoffs_history.append(0)
                imitation_cooperation_rate_history.append(0)
                imitation_avg_degree_history.append(0)
                imitation_avg_trust_history.append(0)

            for i in self.drl_nodes:

                state = self.current_drl_states[i]
                action = self.G.nodes[i]["current_strategy"]
                reward = self.G.nodes[i]["current_payoff"]

                next_state = self.prepare_state(i, t + self.pre_train_epochs)
                self.drl_agent.store_experience(state, action, reward, next_state)
                self.current_drl_states[i] = next_state

            self.drl_agent.train()

            if self.save_checkpoints:

                should_save = False

                if t in self.checkpoint_intervals:
                    should_save = True

                if self.checkpoint_frequency and t % self.checkpoint_frequency == 0:
                    should_save = True

                if should_save:
                    self._save_checkpoint(t, phase="train")

            if t % 500 == 0:
                print(
                    f"Training progress: {t }/{self .train_epochs }, elapsed: {time .perf_counter ()-t1 :.2f}s"
                )
                t1 = time.perf_counter()

        if self.save_checkpoints:
            self._save_checkpoint(self.train_epochs, phase="train_final")

        return (
            payoffs_history,
            cooperation_rate_history,
            avg_degree_history,
            avg_trust_history,
            clustering_history,
            assortativity_history,
            largest_component_history,
            drl_payoffs_history,
            drl_cooperation_rate_history,
            drl_avg_degree_history,
            drl_avg_trust_history,
            imitation_payoffs_history,
            imitation_cooperation_rate_history,
            imitation_avg_degree_history,
            imitation_avg_trust_history,
        )

    def test(self):
        """Execute the test routine for review reproduction."""
        print(f"Starting the test phase; rounds:{self .test_epochs }")

        payoffs_history = []
        cooperation_rate_history = []
        avg_degree_history = []
        avg_trust_history = []
        clustering_history = []
        assortativity_history = []
        largest_component_history = []

        drl_payoffs_history = []
        drl_cooperation_rate_history = []
        drl_avg_degree_history = []
        drl_avg_trust_history = []

        imitation_payoffs_history = []
        imitation_cooperation_rate_history = []
        imitation_avg_degree_history = []
        imitation_avg_trust_history = []

        t1 = time.perf_counter()
        for t in range(1, self.test_epochs + 1):

            self.apply_perturbation(t + self.pre_train_epochs + self.train_epochs)

            # behavior dynamics
            payoffs = self.calculate_payoffs()

            self.store_previous_strategy()

            self.new_neighbors = {i: [] for i in self.G.nodes()}
            for i in self.G.nodes():
                self.update_trust(i)

            for j in self.G.nodes():
                self.update_neighbors(j, t + self.pre_train_epochs + self.train_epochs)

            for i in self.G.nodes():
                if i in self.drl_nodes:

                    self.update_strategy_by_drl(i, is_testing=True)

                    self.prepare_state(i, t + self.pre_train_epochs + self.train_epochs)
                else:
                    self.update_strategy_by_imitation(i)

            self.store_history_data()

            avg_payoffs = np.mean(payoffs)
            payoffs_history.append(avg_payoffs)
            cooperation_rate_history.append(
                np.mean([self.G.nodes[i]["current_strategy"] for i in self.G.nodes()])
            )
            avg_degree_history.append(np.mean([d for _, d in self.G.degree()]))
            avg_trust_history.append(
                np.mean([self.G.nodes[i]["trust"] for i in self.G.nodes()])
            )

            clustering = nx.average_clustering(self.G)
            clustering_history.append(clustering)

            for node in self.G.nodes():
                self.G.nodes[node]["node_type"] = (
                    "drl" if node in self.drl_nodes else "imitation"
                )
            assortativity = nx.attribute_assortativity_coefficient(self.G, "node_type")
            assortativity_history.append(assortativity)

            if nx.is_connected(self.G):
                largest_component_size = self.G.number_of_nodes()
            else:
                largest_component = max(nx.connected_components(self.G), key=len)
                largest_component_size = len(largest_component)
            largest_component_history.append(largest_component_size)

            if len(self.drl_nodes) > 0:
                drl_payoffs = [payoffs[i] for i in self.drl_nodes]
                drl_strategies = [
                    self.G.nodes[i]["current_strategy"] for i in self.drl_nodes
                ]
                drl_degrees = [d for i, d in self.G.degree() if i in self.drl_nodes]
                drl_trusts = [self.G.nodes[i]["trust"] for i in self.drl_nodes]

                drl_payoffs_history.append(np.mean(drl_payoffs) if drl_payoffs else 0)
                drl_cooperation_rate_history.append(
                    np.mean(drl_strategies) if drl_strategies else 0
                )
                drl_avg_degree_history.append(
                    np.mean(drl_degrees) if drl_degrees else 0
                )
                drl_avg_trust_history.append(np.mean(drl_trusts) if drl_trusts else 0)
            else:
                drl_payoffs_history.append(0)
                drl_cooperation_rate_history.append(0)
                drl_avg_degree_history.append(0)
                drl_avg_trust_history.append(0)

            if len(self.imitation_nodes) > 0:
                imitation_payoffs = [payoffs[i] for i in self.imitation_nodes]
                imitation_strategies = [
                    self.G.nodes[i]["current_strategy"] for i in self.imitation_nodes
                ]
                imitation_degrees = [
                    d for i, d in self.G.degree() if i in self.imitation_nodes
                ]
                imitation_trusts = [
                    self.G.nodes[i]["trust"] for i in self.imitation_nodes
                ]

                imitation_payoffs_history.append(
                    np.mean(imitation_payoffs) if imitation_payoffs else 0
                )
                imitation_cooperation_rate_history.append(
                    np.mean(imitation_strategies) if imitation_strategies else 0
                )
                imitation_avg_degree_history.append(
                    np.mean(imitation_degrees) if imitation_degrees else 0
                )
                imitation_avg_trust_history.append(
                    np.mean(imitation_trusts) if imitation_trusts else 0
                )
            else:
                imitation_payoffs_history.append(0)
                imitation_cooperation_rate_history.append(0)
                imitation_avg_degree_history.append(0)
                imitation_avg_trust_history.append(0)

            if t % 500 == 0:
                print(
                    f"Test progress: {t }/{self .test_epochs }, elapsed: {time .perf_counter ()-t1 :.2f}s"
                )
                t1 = time.perf_counter()

        if self.save_checkpoints:
            self._save_checkpoint(self.test_epochs, phase="test_final")

        return (
            payoffs_history,
            cooperation_rate_history,
            avg_degree_history,
            avg_trust_history,
            clustering_history,
            assortativity_history,
            largest_component_history,
            drl_payoffs_history,
            drl_cooperation_rate_history,
            drl_avg_degree_history,
            drl_avg_trust_history,
            imitation_payoffs_history,
            imitation_cooperation_rate_history,
            imitation_avg_degree_history,
            imitation_avg_trust_history,
        )


def run_simulation(abm_model):
    print("Starting pre-training phase.")
    pre_train_results = abm_model.pre_train()
    print("Starting training phase.")
    train_results = abm_model.train()
    print("Starting testing phase.")
    test_results = abm_model.test()
    simulation_results = {
        "pre_train": dict(
            zip(
                [
                    "payoffs",
                    "cooperation_rate",
                    "avg_degree",
                    "avg_trust",
                    "clustering",
                    "assortativity",
                    "largest_component",
                ],
                pre_train_results,
            )
        ),
        "train": {
            "all": dict(
                zip(
                    [
                        "payoffs",
                        "cooperation_rate",
                        "avg_degree",
                        "avg_trust",
                        "clustering",
                        "assortativity",
                        "largest_component",
                    ],
                    train_results[:7],
                )
            ),
            "drl": dict(
                zip(
                    ["payoffs", "cooperation_rate", "avg_degree", "avg_trust"],
                    train_results[7:11],
                )
            ),
            "imitation": dict(
                zip(
                    ["payoffs", "cooperation_rate", "avg_degree", "avg_trust"],
                    train_results[11:],
                )
            ),
        },
        "test": {
            "all": dict(
                zip(
                    [
                        "payoffs",
                        "cooperation_rate",
                        "avg_degree",
                        "avg_trust",
                        "clustering",
                        "assortativity",
                        "largest_component",
                    ],
                    test_results[:7],
                )
            ),
            "drl": dict(
                zip(
                    ["payoffs", "cooperation_rate", "avg_degree", "avg_trust"],
                    test_results[7:11],
                )
            ),
            "imitation": dict(
                zip(
                    ["payoffs", "cooperation_rate", "avg_degree", "avg_trust"],
                    test_results[11:],
                )
            ),
        },
    }
    return simulation_results


def get_experiment_configs():
    """Execute the get experiment configs routine for review reproduction."""

    simple_agent_config = {
        "input_dim": 3,
        "hidden_dim": 16,
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 50,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 2000,
        "grad_clip_norm": 1.0,
    }

    medium_agent_config = {
        "input_dim": 7,
        "hidden_dim": 16,
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 50,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 2000,
        "grad_clip_norm": 1.0,
    }

    complex_agent_config = {
        "history_dim": 2,
        "gru_hidden_dim": 16,
        "gnn_hidden": 16,
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 50,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 2000,
        "grad_clip_norm": 1.0,
    }

    base_config = {
        "n": 1000,
        "c": 0.05,
        "alpha": 2,
        "p_triadic": 0.5,
        "omega": 0.5,
        "epsilon": 0.005,
        "mu": 0.1,
        "rho": 0.1,
        "trust_strategy": "Open",
        "drl_agent_ratio": 0.1,
        "pre_train_epochs": 1000,
        "train_epochs": 3000,
        "test_epochs": 3000,
        "max_memory": 5,
    }

    environment_configs = {
        "ec_1": {
            "payoffs": (1.0, 0.0, 1.5, 0.1),
            "environment_complexity": 1,
            "dynamic_payoff": False,
        },
        "ec_2": {
            "payoffs": (1.0, -0.3, 1.8, 0.1),
            "environment_complexity": 2,
            "dynamic_payoff": False,
        },
        "ec_3": {
            "payoffs": (1.0, -0.6, 2.2, 0.1),
            "environment_complexity": 3,
            "dynamic_payoff": True,
            "perturbation_interval": 500,
            "trust_disturbance_rate": 0.1,
        },
    }

    input_configs = {
        "ic_1": {"input_complexity": "simple", "dqn_agent_params": simple_agent_config},
        "ic_2": {"input_complexity": "medium", "dqn_agent_params": medium_agent_config},
        "ic_3": {
            "input_complexity": "complex",
            "dqn_agent_params": complex_agent_config,
        },
    }

    all_configs = {}
    for ec, env_config in environment_configs.items():
        for ic, input_config in input_configs.items():
            all_configs[f"{ec }&{ic }"] = {**base_config, **env_config, **input_config}

    return all_configs


if __name__ == "__main__":
    PROJECT_PATHS.ensure_output_dirs()
    configs = get_experiment_configs()
    key = "ec_1&ic_1"
    params_dict = configs[key].copy()
    params_dict.update(
        {
            "n": 50,
            "pre_train_epochs": 5,
            "train_epochs": 5,
            "test_epochs": 5,
            "drl_agent_ratio": 0.1,
        }
    )

    set_seed(0)
    model = CooperationABM(**params_dict)
    experiment_results = run_simulation(model)
    result_path = PROJECT_PATHS.results_dir / "smoke_ec_1_ic_1_seed0.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(experiment_results, f)
    print(f"Smoke run completed: {result_path }")
