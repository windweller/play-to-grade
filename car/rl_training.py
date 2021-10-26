from collections import deque
import numpy as np
from tqdm import tqdm

import torch
from car.classifiers import compute_vae_loss

# Contrastive HoareLSTM
def train_dqn(agent, bug_env, lstm, bug_lstm, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              action_input=False, use_grader=True, hoare_threshold=0.1,
              cuda=False):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start  # initialize epsilon

    hiddens, bug_hiddens = None, None

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in tqdm(range(n_episodes)):
        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                inp = state if not action_input else (state, action)
                y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
                bug_y_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

                dist_to_correct = lstm.get_distance(y_hat, next_state)
                dist_to_broken = bug_lstm.get_distance(bug_y_hat, next_state)

                if np.abs(dist_to_correct - dist_to_broken) < hoare_threshold:
                    reward = 0  # 0 means correct
                else:
                    reward = np.argmin([dist_to_correct, dist_to_broken])

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(dist_to_correct - dist_to_broken)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists)
        traj_labels.append(labels)
        traj_preds.append(preds)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, traj_dists, traj_preds, traj_labels

def train_dqn_sae(agent, bug_env, lstm, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              action_input=False, use_grader=True, threshold_offset=0.1,
              cuda=False):

    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    hiddens, bug_hiddens = None, None

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in tqdm(range(n_episodes)):
        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # Contrastive Grader
                # vs. Memorization Grader
                inp = state if not action_input else (state, action)
                y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

                dist_to_correct = lstm.get_distance(y_hat, next_state)
                reward = float(dist_to_correct > threshold_offset)

                # if np.abs(dist_to_correct - dist_to_broken) < hoare_threshold:
                #     reward = 0  # 0 means correct
                # else:
                #     reward = np.argmin([dist_to_correct, dist_to_broken])

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(dist_to_correct)  # dist_to_broken

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists)
        traj_labels.append(labels)
        traj_preds.append(preds)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, traj_dists, traj_preds, traj_labels

def train_dqn_vae(agent, bug_env, model, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              use_grader=True, vae_threshold=0.1,
              cuda=False, history_window=4):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in tqdm(range(n_episodes)):
        state_buffer = deque(maxlen=history_window)

        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # VAE loss (is itself a distance)
                # when we didn't have enough history buffer
                # we just assign 0 to everything
                if len(state_buffer) == history_window:
                    feat_X = np.concatenate(state_buffer)
                    vae_loss = compute_vae_loss(model, torch.from_numpy(feat_X).float(), cuda=cuda)
                    reward = vae_loss > vae_threshold
                else:
                    reward = 0
                    vae_loss = 0

                state_buffer.append(state)

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(vae_loss)  # dist_to_broken

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists)
        traj_labels.append(labels)
        traj_preds.append(preds)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, traj_dists, traj_preds, traj_labels

def train_dqn_gmm(agent, bug_env, model, normalizer, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              use_grader=True, gmm_threshold=0.1,
              cuda=False, history_window=4):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in tqdm(range(n_episodes)):
        state_buffer = deque(maxlen=history_window)

        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # VAE loss (is itself a distance)
                # when we didn't have enough history buffer
                # we just assign 0 to everything
                if len(state_buffer) == history_window:
                    feat_X = np.concatenate(state_buffer)
                    # vae_loss = compute_vae_loss(model, torch.from_numpy(feat_X).float(), cuda=cuda)
                    feat_X = normalizer.transform(np.array([feat_X]))
                    test_y_hat = np.squeeze(model.predict(feat_X))
                    reward = test_y_hat
                else:
                    reward = 0
                    test_y_hat = 0

                state_buffer.append(state)

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(test_y_hat)  # dist_to_broken

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists)
        traj_labels.append(labels)
        traj_preds.append(preds)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, traj_dists, traj_preds, traj_labels

def train_dqn_mlp(agent, bug_env, model, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              use_grader=True,
              cuda=False, history_window=4):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
    # let's look at the predictions made and their quality
    traj_labels, traj_preds = [], []

    for i_episode in tqdm(range(n_episodes)):
        state_buffer = deque(maxlen=history_window)

        state = bug_env.reset()
        score, true_reward = 0, 0
        dists, preds, labels = [], [], []
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = bug_env.step(action)

            if not use_grader:
                reward = 1 if info['bug_state'] else 0
            else:
                # VAE loss (is itself a distance)
                # when we didn't have enough history buffer
                # we just assign 0 to everything
                if len(state_buffer) == history_window:
                    feat_X = np.concatenate(state_buffer)
                    test_y_hat = np.squeeze(model.predict(np.array([feat_X])))
                    reward = test_y_hat
                else:
                    reward = 0
                    test_y_hat = 0

                state_buffer.append(state)

                preds.append(reward)
                labels.append(info['bug_state'])

                dists.append(test_y_hat)  # dist_to_broken

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        traj_dists.append(dists)
        traj_labels.append(labels)
        traj_preds.append(preds)

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    return scores, true_rewards, traj_dists, traj_preds, traj_labels

def compute_perc_traj_with_bug(traj_labels):
    cnt = 0
    for labels in traj_labels:
        if sum(labels) > 0:
            cnt += 1
    return cnt / len(traj_labels)

def collect_trajs(agent, env, lstm, bug_lstm, n_episodes=150, max_t=30,
                  action_input=True, hoare_threshold=0.1,
                  cuda=False, eps=0):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        score, true_reward = 0, 0
        hiddens, bug_hiddens = None, None
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = env.step(action)

            # Contrastive Grader
            # vs. Memorization Grader
            inp = state if not action_input else (state, action)
            y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
            bug_y_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

            dist_to_correct = lstm.get_distance(y_hat, next_state)
            dist_to_broken = bug_lstm.get_distance(bug_y_hat, next_state)

            if np.abs(dist_to_correct - dist_to_broken) < hoare_threshold:
                reward = 0  # 0 means correct
            else:
                reward = np.argmin([dist_to_correct, dist_to_broken])

            state = next_state
            score += reward
            true_reward += 1 if info['bug_state'] else 0
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

        true_reward_window.append(true_reward)
        true_rewards.append(np.mean(true_reward_window))

    return scores, true_rewards