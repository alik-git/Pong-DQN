import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from utils import gym_wrappers
from utils import model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


TransitionTable = collections.namedtuple('TransitionTable',
                                         field_names=['state', 'action', 'reward', 'done', 'new_state'])

# Class to handle storing and randomly sampling state transitions


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(
            len(self.buffer), ("batch_size"), replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx]
                                                             for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


# Class that chooses actions using a DQN policy to step through the environment
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def step_env(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # choose random action every once in a while
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # get possible actions
            state_actions = np.array([self.state], copy=False)
            state_vector = torch.tensor(state_actions).to(device)
            # use network to predict values of possible actions
            q_values_vector = net(state_vector)
            # pick the action with the best value
            _, action_value = torch.max(q_values_vector, dim=1)
            action = int(action_value.item())

        # step in the environment using the action chosen above
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # save transition as experience to experience buffer
        exp_tuple = TransitionTable(self.state, action, reward, is_done,
                                    new_state)
        self.exp_buffer.append(exp_tuple)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def compute_loss(batch, net, target_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # convert everything into a torch tensor so that we can compute derivatives
    states_vector = torch.tensor(states).to(device)
    next_states_vector = torch.tensor(next_states).to(device)
    actions_vector = torch.tensor(actions).to(device)
    rewards_vector = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # compute the predictions our model would make for this batch of state
    # transitions (represented as tensors)
    state_action_values = net(states_vector).gather(1,
                                                    actions_vector.unsqueeze(-1)).squeeze(-1)
    # compute the actual values we got for those transitions
    next_state_values = target_net(next_states_vector).max(1)[0]
    # make sure future values aren't being considered for end states
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    # add the future reward of a state reached to the rewards of the current
    # transition to correctly represent the actual value of the state reached
    expected_state_action_values = next_state_values * GAMMA + rewards_vector

    # compute the Mean Squared Error Loss between our predictions and the
    # actual values of the transitions
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    # handle arguments to the python file
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default= %.2f" %
                        MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym_wrappers.make_env(args.env)

    # make two networks, one for local training, and update the target network
    # to it every once in a while. If you find this confusing you're not alone,
    # you just need to do some reading and googling and you'll get it with time
    local_model = model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    target_model = model.DQN(env.observation_space.shape,
                             env.action_space.n).to(device)

    # log all your results, ideally in a way thats easy to visualize
    logs = SummaryWriter(comment="-" + args.env)
    print(local_model)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    # adam optimizer, just a fancier gradient descent algorithm
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    curr_frame_idx = 0
    prev_frame_idx = 0
    curr_time = time.time()
    best_mean_reward = None

    while True:
        curr_frame_idx += 1

        # probability of choosing a random action at this step
        epsilon = max(EPSILON_FINAL, EPSILON_START - curr_frame_idx /
                      EPSILON_DECAY_LAST_FRAME)

        # step in the environment
        reward = agent.step_env(local_model, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (curr_frame_idx - prev_frame_idx) / \
                (time.time() - curr_time)
            prev_frame_idx = curr_frame_idx
            curr_time = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" %
                  (curr_frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            logs.add_scalar("epsilon", epsilon, curr_frame_idx)
            logs.add_scalar("speed", speed, curr_frame_idx)
            logs.add_scalar("reward_100", mean_reward, curr_frame_idx)
            logs.add_scalar("reward", reward, curr_frame_idx)

            # if the model is has shown consistent improvement, save the model
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(local_model.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved"
                          % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % curr_frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if curr_frame_idx % SYNC_TARGET_FRAMES == 0:
            target_model.load_state_dict(local_model.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = compute_loss(batch, local_model, target_model, device=device)
        loss_t.backward()
        optimizer.step()
    logs.close()
