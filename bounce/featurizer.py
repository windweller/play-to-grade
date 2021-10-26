"""
We turn all previous models into featurizers!
Input a new MDP, we return features
"""

from tqdm import tqdm

import numpy as np
import torch

from collections import deque
import torch.optim as optim

from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from bounce.classifiers import VAE, train_vae_one_epoch, compute_vae_loss, HoareLSTM, train_lstm_one_epoch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

from collections import deque

def hoarelstm_featurize(agent, bug_env, lstm, n_episodes=10,
                                      eps_start=0.05, eps_end=0.01, eps_decay=0.995, max_t=30,
                                      action_input=False, use_grader=True, offset_threshold=0.1,
                                      cuda=False, alpha=1., beta=1.):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    hiddens, bug_hiddens = None, None

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []

    # features
    traj_correct_state_dists, traj_correct_reward_dists = [], []
    traj_orig_rewards = []

    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in range(n_episodes):
        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        dist_to_corrects, dist_to_brokens = [], []
        dist_to_corrects_rew, dist_to_brokens_rew = [], []
        orig_rewards = []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)
            orig_rewards.append(reward)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # Contrastive Grader
                # vs. Memorization Grader
                inp = state if not action_input else (state, action)
                y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

                dist_to_correct = alpha * lstm.get_distance(y_hat, state)

                dist_to_corrects.append(dist_to_correct)

                dist_to_correct_rew = beta * lstm.get_reward_distance(rew_hat, reward)

                dist_to_corrects_rew.append(dist_to_correct_rew)

                dist_to_correct += dist_to_correct_rew

                if dist_to_correct < offset_threshold:
                    reward = 0  # 0 means correct
                else:
                    reward = 1

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(dist_to_correct)

            # agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists); traj_labels.append(labels); traj_preds.append(preds)

        traj_correct_state_dists.append(dist_to_corrects)
        traj_correct_reward_dists.append(dist_to_corrects_rew)

        traj_orig_rewards.append(orig_rewards)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, np.array([sum(l) for l in traj_orig_rewards]), np.array(
        [sum(l) for l in traj_correct_state_dists]), np.array([sum(l) for l in traj_correct_reward_dists])


# n_episodes is K
def contrastive_hoarelstm_featurize(agent, bug_env, lstm, bug_lstm, n_episodes=10,
                                      eps_start=0.05, eps_end=0.01, eps_decay=0.995, max_t=30,
                                      action_input=False, use_grader=True, hoare_threshold=0.1,
                                      cuda=False, alpha=1., beta=1.):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    hiddens, bug_hiddens = None, None

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []

    # features
    traj_correct_state_dists, traj_correct_reward_dists = [], []
    traj_broken_state_dists, traj_broken_reward_dists = [], []
    traj_orig_rewards = []

    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in range(n_episodes):
        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        dist_to_corrects, dist_to_brokens = [], []
        dist_to_corrects_rew, dist_to_brokens_rew = [], []
        orig_rewards = []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)
            orig_rewards.append(reward)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # Contrastive Grader
                # vs. Memorization Grader
                inp = state if not action_input else (state, action)
                y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
                bug_y_hat, bug_rew_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

                dist_to_correct = alpha * lstm.get_distance(y_hat, state)
                dist_to_broken = alpha * bug_lstm.get_distance(bug_y_hat, state)

                dist_to_corrects.append(dist_to_correct); dist_to_brokens.append(dist_to_broken)

                dist_to_correct_rew = beta * lstm.get_reward_distance(rew_hat, reward)
                dist_to_broken_rew = beta * bug_lstm.get_reward_distance(bug_rew_hat, reward)

                dist_to_corrects_rew.append(dist_to_correct_rew); dist_to_brokens_rew.append(dist_to_broken_rew)

                dist_to_correct += dist_to_correct_rew
                dist_to_broken += dist_to_broken_rew

                if np.abs(dist_to_correct - dist_to_broken) < hoare_threshold:
                    reward = 0  # 0 means correct
                else:
                    reward = np.argmin([dist_to_correct, dist_to_broken])

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(dist_to_correct - dist_to_broken)

            # agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists); traj_labels.append(labels); traj_preds.append(preds)

        traj_correct_state_dists.append(dist_to_corrects); traj_broken_state_dists.append(dist_to_brokens)
        traj_correct_reward_dists.append(dist_to_corrects_rew); traj_broken_reward_dists.append(dist_to_brokens_rew)

        traj_orig_rewards.append(orig_rewards)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, np.array([sum(l) for l in traj_orig_rewards]), np.array([sum(l) for l in traj_correct_state_dists]), \
            np.array([sum(l) for l in traj_broken_state_dists]), np.array([sum(l) for l in traj_correct_reward_dists]), \
            np.array([sum(l) for l in traj_broken_reward_dists])