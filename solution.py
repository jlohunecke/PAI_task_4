import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList(([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]))
        self.hidden_activations = nn.ModuleList([getattr(nn, activation)() for _ in range(hidden_layers)])
        self.output_layer_mean = nn.Linear(hidden_size, output_dim)
        self.output_layer_log_std = nn.Linear(hidden_size, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        x = self.input_layer(s)
        for layer, activation in zip(self.hidden_layers, self.hidden_activations):
            x = activation(layer(x))

        output_mean = self.output_layer_mean(x)
        output_log_std = self.output_layer_log_std(x)

        # Concatenate mean and log_std along the second dimension
        output = torch.cat([output_mean, output_log_std], dim=1)

        return output


class Actor:
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        ##### new params #####
        self.actor = None
        self.optimizer = None
        ######################

        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py.
        actor_nn = NeuralNetwork(self.state_dim, self.action_dim, self.hidden_size, self.hidden_layers, 'ReLU')
        self.actor = actor_nn.to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'

        ## added by me #######
        state = state.unsqueeze(0) if state.shape == (3,) else state
        ######################

        action, log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        output = self.actor(state)

        if deterministic:
            action = output[:, 0]
            log_prob = torch.zeros(state.shape[0])
        elif not deterministic:
            log_std = self.clamp_log_std(output[:, 1])
            std = torch.exp(log_std)
            mean = output[:, 0]
            normal = Normal(mean, std)
            action = normal.rsample()
            log_prob = normal.log_prob(action)

        action = action.unsqueeze(1) if (action.shape == (state.shape[0],) and state.shape[0] != 1) else action
        log_prob = log_prob.unsqueeze(1) if (log_prob.shape == (state.shape[0],) and state.shape[0] != 1) else log_prob

        action = action.clamp(-1, 1)

        assert (action.shape == (self.action_dim,) and
                log_prob.shape == (self.action_dim,)) or (action.shape == (state.shape[0], 1) and
                                                          log_prob.shape == (
                                                              state.shape[0],
                                                              1)), 'Incorrect shape for action or log_prob.'
        return action, log_prob

# changed critic_lr: int to critic_lr: float below
class Critic:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, critic_lr: float, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        ##### new params #####
        self.critic_1 = None
        self.critic_2 = None
        self.optimizer = None
        ######################

        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic_1 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'ReLU')
        self.critic_2 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'ReLU')
        self.critic_1 = self.critic_1.to(self.device)
        self.critic_2 = self.critic_2.to(self.device)
        self.optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                    lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''

    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        ##### new params #####
        self.critic_base = None
        self.critic_target = None
        self.policy = None
        self.hidden_size = 256
        self.hidden_layers = 2
        self.critic_lr = 0.001
        self.discount = 0.99
        self.tau = 0.005
        self.temperature = TrainableParameter(0.2, 0.001, True, self.device)
        self.target_entropy = -1 * self.action_dim
        ######################

        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        self.critic_base = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim, self.action_dim,
                                  self.device)
        self.critic_target = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim,
                                    self.action_dim, self.device)
        self.policy = Actor(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim, self.action_dim,
                            self.device)

        self.critic_target_update(self.critic_base.critic_1, self.critic_target.critic_1, 0.0, False)
        self.critic_target_update(self.critic_base.critic_2, self.critic_target.critic_2, 0.0, False)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action = np.random.uniform(-1, 1, (1,))
        action = self.policy.get_action_and_log_prob(torch.tensor(s, dtype=torch.float32, device=self.device),
                                                     deterministic=train)[0].detach().cpu().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # TODO: Implement Critic(s) update here.
        with torch.no_grad():
            next_action, next_log_prob = self.policy.get_action_and_log_prob(s_prime_batch, True)
            next_q1 = self.critic_target.critic_1(torch.cat((s_prime_batch, next_action), 1))[0]
            next_q2 = self.critic_target.critic_2(torch.cat((s_prime_batch, next_action), 1))[0]
            next_q = torch.min(next_q1, next_q2) - self.temperature.get_param() * next_log_prob
            target_q = r_batch + self.discount * next_q

        q1 = self.critic_base.critic_1(torch.cat((s_batch, a_batch), 1))[0]
        q2 = self.critic_base.critic_2(torch.cat((s_batch, a_batch), 1))[0]
        critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)
        self.run_gradient_update_step(self.critic_base, critic_loss)

        # TODO: Implement Policy update here
        action, log_prob = self.policy.get_action_and_log_prob(s_batch, True)
        q1 = self.critic_base.critic_1(torch.cat((s_batch, action), 1))[0]
        q2 = self.critic_base.critic_2(torch.cat((s_batch, action), 1))[0]
        q = torch.min(q1, q2)
        policy_loss = (self.temperature.get_param() * log_prob - q).mean()
        self.run_gradient_update_step(self.policy, policy_loss)

        self.critic_target_update(self.critic_base.critic_1, self.critic_target.critic_1, self.tau, True)
        self.critic_target_update(self.critic_base.critic_2, self.critic_target.critic_2, self.tau, True)

        temp_loss = -(self.temperature.get_log_param() * (-log_prob + self.target_entropy).detach()).mean()
        self.temperature.optimizer.zero_grad()
        temp_loss.backward()
        self.temperature.optimizer.step()

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
