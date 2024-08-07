import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import collections

env = gym.make("CartPole-v1")
class ConfigArgs:
    beta = 0.2
    lamda = 0.1
    eta = 100.0 # scale factor for intrinsic reward
    discounted_factor = 0.99
    lr_critic = 0.005
    lr_actor = 0.001
    lr_icm = 0.001
    max_eps = 500
    sparse_mode = True

args = ConfigArgs()

def to_tensor(x, dtype=None):
    return torch.tensor(x, dtype=dtype).unsqueeze(0)

def get_action_from_probabilities(action_probs):
    return np.random.choice(len(action_probs), 1, p=action_probs)[0]


class Actor(nn.Module):
    def __init__(self, n_actions, space_dims, hidden_dims):
        super(Actor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims, n_actions),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        policy = self.actor(features)
        return policy


class Critic(nn.Module):
    '''Simplified Critic, does not consider the action take
    to get to the specific state, x)'''
    
    def __init__(self, space_dims, hidden_dims):
        super(Critic, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True)
        )
        
        self.critic = nn.Linear(hidden_dims, 1)
    def forward(self, x):
        features = self.feature_extractor(x)
        est_reward = self.critic(features)
        return est_reward
    

actor = Actor(n_actions=env.action_space.n, space_dims=4, hidden_dims=32)
critic = Critic(space_dims=4, hidden_dims=32)


a_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
c_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

class InverseModel(nn.Module):
    '''
    Input features for s1 and s2 and output the action that transitions between them
    '''
    def __init__(self, n_actions, hidden_dims):
        super(InverseModel, self).__init__()
        self.fc = nn.Linear(hidden_dims*2, n_actions)
    def forward(self, features):
        features = features.reshape(1, -1)
        recreated_action_prob = self.fc(features)
        return recreated_action_prob # ToDO: should I do softmax here?

class ForwardModel(nn.Module):
    '''
    Input (s1_feature, action) and output the feature of the next state
    '''
    def __init__(self, n_actions, hidden_dims):
        super(ForwardModel, self).__init__()
        self.fc = nn.Linear(n_actions + hidden_dims, hidden_dims)
        self.eye_mat = torch.eye(n_actions)

    def forward(self, action, features):
        action_rep = self.eye_mat[action]
        x = torch.cat([features, action_rep], dim=-1) 
        features = self.fc(x)
        return features # TODO: should I do a tanh here?

class StateFeatureExtractor(nn.Module):
    '''
    Extracts a feature vector from a state
    '''
    def __init__(self, space_dims, hidden_dims):
        super(StateFeatureExtractor, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y

feature_extractor = StateFeatureExtractor(env.observation_space.shape[0], 32)
forward_model = ForwardModel(env.action_space.n, 32)
inverse_model = InverseModel(env.action_space.n, 32)

icm_params = list(feature_extractor.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters())
icm_optim = torch.optim.Adam(icm_params, lr=args.lr_icm)

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
    
    def forward(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss

pg_loss = PGLoss()
mse_loss = nn.MSELoss()
xe_loss = nn.CrossEntropyLoss()

global_step =0
n_eps = 0
reward_list = []
mva_lst = []
mva = 0.
avg_ireward_lst = []

while n_eps < args.max_eps:
    n_eps += 1
    next_obs = to_tensor(env.reset()[0], dtype=torch.float)
    done = False
    score = 0
    intrinsic_reward_list = []

    while not done:
        obs = next_obs
        a_optim.zero_grad()
        c_optim.zero_grad()
        icm_optim.zero_grad()

        # get action from the actor
        policy = actor(obs) # action probabilities
        action = get_action_from_probabilities(policy.detach().numpy()[0])

        # Take step in the env
        next_obs, extrinsic_reward, done, _, info = env.step(action)
        next_obs = to_tensor(next_obs, dtype=torch.float)
        advantages = torch.zeros_like(policy)
        if args.sparse_mode:
            extrinsic_reward = to_tensor([0.], dtype=torch.float)
        else:
            extrinsic_reward = to_tensor([extrinsic_reward], dtype=torch.float)
        t_action = to_tensor(action)

        # Run the critic
        curr_v = critic(obs)[0]
        next_v = critic(next_obs)[0]

        # ICM
        obs_stacked = torch.cat([obs, next_obs], dim=0)
        features_stacked = feature_extractor(obs_stacked)
        inverse_action_prob = inverse_model(features_stacked)
        est_next_features = forward_model(t_action, features_stacked[0:1])

        forward_loss = mse_loss(est_next_features, features_stacked[1])
        inverse_loss = xe_loss(inverse_action_prob, t_action.view(-1))
        icm_loss = (1 - args.beta)*inverse_loss + args.beta*forward_loss

        # Reward
        intrinsic_reward = args.eta*forward_loss.detach() # How bad am I at predicting my next state
        total_reward = intrinsic_reward + extrinsic_reward
        if done:
            advantages[0, action] = total_reward - curr_v
            c_target = total_reward
        else:
            advantages[0, action] = total_reward + args.discounted_factor*next_v - curr_v
            c_target = total_reward + args.discounted_factor*next_v

        # Loss - Actor/Critic
        actor_loss = pg_loss(policy, advantages.detach())
        critic_loss = mse_loss(curr_v, c_target.detach())
        ac_loss = actor_loss + critic_loss

        # Total loss and update step
        loss = args.lamda*ac_loss + icm_loss
        loss.backward()
        icm_optim.step()
        a_optim.step()
        c_optim.step()

        
        if not done:
            score += 1


        intrinsic_reward_list.append(intrinsic_reward.item())

        global_step+=1

    avg_intrinsic_reward = np.sum(np.array(intrinsic_reward_list)) / len(intrinsic_reward_list)
    mva = 0.95*mva + 0.05*score
    avg_ireward_lst.append(avg_intrinsic_reward)
    mva_lst.append(mva)
    reward_list.append(score)
    print(f"Episode: {n_eps}, Score: {score}, AVG Score: {mva}, AVG Intrinsic Reward: {avg_intrinsic_reward}")

env.close()