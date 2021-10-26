from collections import deque
import numpy as np
from tqdm import tqdm

def train_dqn(agent, env, n_episodes=150, eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              cuda=False, bug_reward=False, reward_scale=1.0):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    training_done = False

    pbar = tqdm(range(n_episodes))
    actions = []

    for i_episode in range(n_episodes):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            if bug_reward:
                reward = 1 if info['bug_state'] else 0
                reward *= reward_scale

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        pbar.update(1)
        pbar.set_postfix({'reward': np.mean(scores_window)})

    return scores, actions


def train_collaborative_dqn(agent, bug_env, lstm, bug_lstm, n_episodes=150,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=30,
              action_input=False, use_grader=True, hoare_threshold=0.1,
              cuda=False, alpha=1, beta=1):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    hiddens, bug_hiddens = None, None

    true_rewards = []
    true_reward_window = deque(maxlen=100)

    # implement this new metric
    traj_dists = []
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
                y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
                bug_y_hat, bug_rew_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

                dist_to_correct = alpha * lstm.get_distance(y_hat, state)
                dist_to_broken = alpha * bug_lstm.get_distance(bug_y_hat, state)

                dist_to_correct += beta * lstm.get_reward_distance(rew_hat, reward)
                dist_to_broken += beta * bug_lstm.get_reward_distance(bug_rew_hat, reward)

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

def compute_perc_traj_with_bug(traj_labels):
    cnt = 0
    for labels in traj_labels:
        if sum(labels) > 0:
            cnt += 1
    return cnt / len(traj_labels)

def collect_trajs(agent, env, lstm, bug_lstm, n_episodes=150, max_t=30,
                  action_input=True, hoare_threshold=0.1,
                  cuda=False, eps=0, alpha=1, beta=1):
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

            inp = state if not action_input else (state, action)
            y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
            bug_y_hat, bug_rew_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

            dist_to_correct = alpha * lstm.get_distance(y_hat, state)
            dist_to_broken = alpha * bug_lstm.get_distance(bug_y_hat, state)

            dist_to_correct += beta * lstm.get_reward_distance(rew_hat, reward)
            dist_to_broken += beta * bug_lstm.get_reward_distance(bug_rew_hat, reward)

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

def evaluate_agent(agent, env, n_episodes=50, max_t=150, cuda=True, eps=0):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    true_rewards = []
    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        score, true_reward = 0, 0
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward
            if done:
                break

        true_rewards.append(score)
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score (episode score)

    return scores, true_rewards