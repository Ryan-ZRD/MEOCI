import torch
import torch.nn as nn

class DuelingQNetwork(nn.Module):


    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 256, 128),
                 activation="relu", dropout=0.1):
        """
        :param state_dim: Dimension of state input (e.g., accuracy, queue_len, resource, task_rate)
        :param action_dim: Dimension of action space (number of partition + exit combinations)
        :param hidden_dims: Tuple defining the MLP hidden layer sizes
        :param activation: Activation function: 'relu' | 'leakyrelu' | 'gelu'
        :param dropout: Dropout probability for regularization
        """
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define activation mapping
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leakyrelu":
            act_fn = nn.LeakyReLU(0.1)
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Feature extractor (shared representation)
        layers = []
        input_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), act_fn, nn.Dropout(dropout)]
            input_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            act_fn,
            nn.Linear(128, 1)  # Outputs scalar V(s)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            act_fn,
            nn.Linear(128, action_dim)  # Outputs A(s, a) for all actions
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through Dueling Q-Network.
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))

        :param state: torch.Tensor of shape (batch_size, state_dim)
        :return: Q-values tensor of shape (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine according to Dueling DQN formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DoubleDuelingDQN(nn.Module):
    """
    Wrapper module maintaining both online and target networks
    ----------------------------------------------------------
    Used for Double DQN updates:
        Q_target ← r + γ * Q_target'(s', argmax_a Q_online(s', a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 256, 128),
                 activation="relu", tau=0.005):
        super(DoubleDuelingDQN, self).__init__()

        self.online_net = DuelingQNetwork(state_dim, action_dim, hidden_dims, activation)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dims, activation)

        # Initialize target with online weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.tau = tau  # Soft update factor

    @torch.no_grad()
    def soft_update(self):
        """Perform soft target network update: θ_target ← τθ_online + (1-τ)θ_target"""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the online network"""
        return self.online_net(state)

    def q_target(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the target network"""
        with torch.no_grad():
            return self.target_net(state)

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str, map_location=None):
        """Load model weights"""
        self.online_net.load_state_dict(torch.load(path, map_location=map_location))
        self.target_net.load_state_dict(torch.load(path, map_location=map_location))
