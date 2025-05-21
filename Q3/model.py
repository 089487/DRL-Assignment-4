import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
import os
import pickle
from tqdm import tqdm

class QContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QContinuous, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)

class PolicyContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, log_std_min=-20.0, log_std_max=2.0):
        super(PolicyContinuous, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state, deterministic=False):
        x = self.fc(state)
        mu = self.mu(x)
        
        # Constrain log_std within [log_std_min, log_std_max]
        raw_log_std = self.log_std(x)

        log_std = torch.clamp(raw_log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # Create normal distribution
        dist = Normal(mu, std)
        
        if deterministic:
            action = torch.tanh(mu)
            log_prob = None
        else:
            # Reparameterization trick
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
            
            # Calculate log_prob
            log_prob = dist.log_prob(x_t)
            
            # Apply tanh squashing correction to log_prob
            log_prob -= torch.log(1-y_t.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            action = y_t
            
        return action, log_prob

class SAC:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_dim=512, 
                 actor_lr=3e-4, 
                 critic_lr=3e-4, 
                 alpha=0.2,
                 alpha_lr=1e-4, 
                 target_entropy=None, 
                 tau=0.005,
                 gamma=0.99,
                 batch_size=256,
                 buffer_size=1000000,
                 initial_fill=50000,
                 log_std_min=-20.0,
                 log_std_max=2.0,
                 device=None):

        #self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.actor = PolicyContinuous(state_dim, hidden_dim, action_dim, log_std_min, log_std_max).to(self.device)
        self.critic_1 = QContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_2 = QContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_1_target = QContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_2_target = QContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Set target networks to eval mode
        self.critic_1_target.eval()
        self.critic_2_target.eval()
        
        # Initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optim_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # Initialize alpha (temperature parameter)
        self.alpha = alpha
        self.log_alpha = torch.tensor(-1.0,dtype=torch.float32).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Set target entropy to -action_dim if not specified
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        
        # Set other parameters
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.initial_fill = initial_fill
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=buffer_size,device = self.device),batch_size=self.batch_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
    def select_action(self, state, deterministic=False, eval_mode=False):
        """Select an action from state."""
        with torch.no_grad():
            if eval_mode:
                self.actor.eval()
            
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state, deterministic)
            
            if eval_mode:
                self.actor.train()
                
            return action.cpu().numpy().flatten()
    
    def cal_target(self, reward, next_state, done):
        """Calculate target Q-value."""
        with torch.no_grad():
            # Sample action from policy
            next_action, next_log_prob = self.actor(next_state)
            
            # Calculate target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            
            # Calculate final target
            target = reward + (1 - done) * self.gamma * target_q
            
        return target
    
    def soft_update(self):
        """Soft update for target networks."""
        # Update target critic 1
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
            
        # Update target critic 2
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        self.buffer.add(
            TensorDict({
                "state": state, 
                "action": action, 
                "reward": reward, 
                "next_state": next_state, 
                "done": done
            }, batch_size=[])
        )
    
    def sample_from_buffer(self):
        """Sample batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None
            
        batch = self.buffer.sample()
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        
        return states, actions, rewards, next_states, dones
    
    def update(self):
        """Update the networks."""
        # Skip update if buffer is not filled enough
        if len(self.buffer) < max(self.batch_size, self.initial_fill):
            return None, None, None
            
        # Sample from buffer
        batch = self.sample_from_buffer()
        if batch is None:
            return None, None, None
            
        states, actions, rewards, next_states, dones = batch
        
        # Make sure tensors are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.view(-1, 1).to(self.device)
        
        # Update critics
        # Calculate target Q value
        with torch.no_grad():
            target = self.cal_target(rewards, next_states, dones)
        
        # Calculate current Q values
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        
        # Calculate critic losses
        critic_loss_1 = F.mse_loss(q1, target)
        critic_loss_2 = F.mse_loss(q2, target)
        
        # Update critic 1
        self.critic_optim_1.zero_grad()
        critic_loss_1.backward()
        # Clip critic grad norm to 1
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1)
        self.critic_optim_1.step()
        
        # Update critic 2
        self.critic_optim_2.zero_grad()
        critic_loss_2.backward()
        # Clip critic grad norm to 1
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1)
        self.critic_optim_2.step()
        
        # Update actor
        # Sample actions from policy
        new_actions, log_probs = self.actor(states)
        
        # Calculate Q values for new actions
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Calculate actor loss
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # Clip actor grad norm to 1
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        
        # clip alpha grad norm to 1
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        
        self.alpha_optim.step()
        
        # Update target networks
        self.soft_update()
        
        # Increment training step
        self.training_step += 1
        
        return critic_loss_1.item() + critic_loss_2.item(), actor_loss.item(), alpha_loss.item()
    
    def save(self, path):
        """Save model parameters."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(path, "critic_1.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(path, "critic_2.pth"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pth"))
        torch.save({
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim_1': self.critic_optim_1.state_dict(),
            'critic_optim_2': self.critic_optim_2.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict()
        }, os.path.join(path, "optimizers.pth"))
        
    def load(self, path):
        """Load model parameters."""
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Please check the path.")
            return False
            
        try:
            self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=self.device))
            self.critic_1.load_state_dict(torch.load(os.path.join(path, "critic_1.pth"), map_location=self.device))
            self.critic_2.load_state_dict(torch.load(os.path.join(path, "critic_2.pth"), map_location=self.device))
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())
            
            self.log_alpha = torch.load(os.path.join(path, "log_alpha.pth"), map_location=self.device)
            self.log_alpha.requires_grad = True
            
            # Load optimizer states if they exist
            optim_path = os.path.join(path, "optimizers.pth")
            if os.path.exists(optim_path):
                optim_states = torch.load(optim_path, map_location=self.device)
                self.actor_optim.load_state_dict(optim_states['actor_optim'])
                self.critic_optim_1.load_state_dict(optim_states['critic_optim_1'])
                self.critic_optim_2.load_state_dict(optim_states['critic_optim_2'])
                self.alpha_optim.load_state_dict(optim_states['alpha_optim'])
                
            print(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
            return False
    
    def save_buffer(self, path):
        """Save replay buffer."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        try:
            self.buffer.save(os.path.join(path, "replay_buffer"))
            print(f"Replay buffer saved to {path}")
            return True
        except Exception as e:
            print(f"Failed to save replay buffer to {path}: {e}")
            return False
    
    def load_buffer(self, path):
        """Load replay buffer."""
        full_path = os.path.join(path, "replay_buffer")
        if not os.path.exists(full_path):
            print(f"Path {full_path} does not exist. Please check the path.")
            return False
            
        try:
            self.buffer.load(full_path)
            print(f"Replay buffer loaded from {full_path} with {len(self.buffer)} transitions")
            return True
        except Exception as e:
            print(f"Failed to load replay buffer from {full_path}: {e}")
            self.buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=1000000))
            print("New replay buffer created.")
            return False

def evaluate_policy(agent, env, num_episodes=10, seed=None):
    """Evaluate policy for multiple episodes."""
    episode_rewards = []
    
    # Set evaluation seed if provided
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=np.random.randint(0, 1000000) if seed is None else seed)
        episode_reward = 0
        done = False
        
        while not done:
            # Select action deterministically
            action = agent.select_action(state, deterministic=True, eval_mode=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update reward and state
            episode_reward += reward
            state = next_state
            
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    return mean_reward, std_reward, min_reward, max_reward

def train_SAC(env_name="humanoid-walk", 
              num_episodes=10000,
              hidden_dim=256,
              actor_lr=3e-4,
              critic_lr=3e-4,
              alpha=0.2,
              alpha_lr=3e-4,
              gamma=0.99,
              tau=0.005,
              batch_size=256,
              buffer_size=1000000,
              initial_fill=10000,
              eval_freq=10,
              eval_episodes=10,
              log_freq=1,
              save_freq=100,
              load_model=True,
              load_buffer=True,
              model_path="trained_model/",
              buffer_path="replay_buffer/",
              log_std_min=-20.0,
              log_std_max=2.0,
              seed=None):
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Make environment
    from dmc import make_dmc_env
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize SAC agent
    sac_agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha=alpha,
        alpha_lr=alpha_lr,
        target_entropy=-action_dim,  # Set target entropy to -action_dim
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        initial_fill=initial_fill,
        log_std_min=log_std_min,
        log_std_max=log_std_max
    )
    
    print(f"Agent initialized with device: {sac_agent.device}")
    
    # Load model if requested
    if load_model:
        sac_agent.load(model_path)
