"""
This file is on a need-base, we only import and change functions that we definitely need
"""

# Training methods are stored here

from tqdm import tqdm

import numpy as np

import torch.optim as optim

from bounce.classifiers import VAE, train_vae_one_epoch, compute_vae_loss, HoareLSTM, train_lstm_one_epoch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ===== Sequential Autoencoder =======

import torch.nn as nn
l1_loss = nn.L1Loss(reduction='mean')

def train_naive_sae_lstm_model(policy, env, bug_env, epochs=100, history_window=4, cuda=False,
                               batch_size=4, delta=True, action_input=True, eps=0.01, threshold_offset=0.5,
                               auto_threshold=False):
    feat_dim = 4

    lstm = HoareLSTM(4, batch_size=batch_size, delta=delta,
                     action_input=action_input)
    if cuda:
        lstm = lstm.to('cuda')

    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)

    test_accu = []
    bug_precision, bug_recall = [], []

    train_losses = []

    for e in tqdm(range(epochs)):

        states, actions = [], []
        state = env.reset()
        states.append(state)
        for i in range(30):
            # replace here with real policy
            action = policy.get_action(state, eps=eps)
            state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            if done:
                break

        states = np.vstack(states)
        actions = np.array(actions)
        if action_input:
            lstm.store(states, actions)
        else:
            lstm.store(states)  # 1 data point in the batch

        for _ in range(5):
            loss = train_lstm_one_epoch(lstm, optimizer, l1_loss, cuda=cuda)
            train_losses.append(np.mean(loss))  # average batch loss

        # get test set
        if e % 5 == 0:
            preds_y, test_y = [], []
            for _ in range(10):
                states = []
                preds = []
                labels = []

                state = bug_env.reset()
                hiddens = None

                for i in range(30):

                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

                    # next-state
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if info['bug_state']:
                        labels.append(1)
                    else:
                        labels.append(0)

                    dist = lstm.get_distance(y_hat, state)
                    delta = 0
                    if auto_threshold:
                        delta = np.mean(train_losses[-10:])
                    delta += threshold_offset
                    preds.append(dist > delta)
                    states.append(state)

                    if done:
                        break

                assert len(preds) == len(labels)
                states = np.vstack(states)
                preds_y.append(np.array(preds))
                test_y.append(np.array(labels))

            test_y = np.concatenate(test_y)
            preds_y = np.concatenate(preds_y)

            accu = accuracy_score(test_y, preds_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, preds_y)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, train_losses

def train_naive_sae_lstm_model_inner(policy, lstm, optimizer,
                                       env, bug_env, epochs=100, history_window=4, threshold_offset=0.01,
                                       correct_training_epoch=5, cuda=False, delta=False, action_input=False,
                                       batch_size=4, test_every=5, eps=0, alpha=1.0, beta=1.0, max_t=200):
    test_accu = []
    bug_precision, bug_recall = [], []
    train_losses, bug_train_losses = [], []

    correct_traj_states, broken_traj_states = [], []

    for e in tqdm(range(epochs)):

        states, actions, rewards = [], [], []
        state = env.reset()
        states.append(state)
        for i in range(max_t):
            action = policy.get_action(state, eps=eps)
            state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                break

        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        if action_input:
            lstm.store(states, actions, rewards)
        else:
            lstm.store(states)  # 1 data point in the batch
        correct_traj_states.append(states)

        if e % 5 == 0:
            train_epochs = correct_training_epoch
            for _ in range(train_epochs):
                loss = train_lstm_one_epoch(lstm, optimizer, l1_loss, cuda=cuda)
                train_losses.append(np.mean(loss))  # average batch loss

        # get test set
        if e % test_every == 0:
            preds_y, test_y = [], []
            for _ in range(10):
                states, actions = [], []
                preds, labels = [], []

                state = bug_env.reset()
                hiddens, bug_hiddens = None, None

                for i in range(max_t):

                    # replace here with real policy
                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

                    # next-state
                    state, reward, done, info = bug_env.step(action)
                    if info['bug_state']:
                        labels.append(1)
                    else:
                        labels.append(0)

                    dist_to_correct = alpha * lstm.get_distance(y_hat, state)
                    dist_to_correct += beta * lstm.get_reward_distance(rew_hat, reward)

                    preds.append(dist_to_correct > threshold_offset)

                    states.append(state)
                    actions.append(action)

                    if done:
                        break

                assert len(preds) == len(labels)
                preds_y.append(np.array(preds))
                test_y.append(np.array(labels))

            test_y = np.concatenate(test_y)
            preds_y = np.concatenate(preds_y)

            accu = accuracy_score(test_y, preds_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, preds_y)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (train_losses, lstm, correct_traj_states, broken_traj_states)

# ======== HoareLSTM ========
def train_naive_hoare_lstm_model(policy, env, bug_env, epochs=100, history_window=4, hoare_threshold=0.01,
                                 fresh_lstm=False, cuda=False, delta=False, action_input=False,
                                 batch_size=4, test_every=5):
    lstm = HoareLSTM(4, batch_size=batch_size, delta=delta,
                     action_input=action_input)
    if cuda:
        lstm = lstm.to('cuda')
    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)

    bug_lstm = HoareLSTM(4, batch_size=batch_size, delta=delta, action_input=action_input)
    if cuda:
        bug_lstm = bug_lstm.to('cuda')
    bug_optimizer = optim.Adam(bug_lstm.parameters(), lr=1e-3)

    return train_naive_hoare_lstm_model_inner(policy, lstm, bug_lstm, optimizer, bug_optimizer,
                                       env, bug_env, epochs, history_window, hoare_threshold,
                                       fresh_lstm, cuda, delta, action_input)


def train_naive_hoare_lstm_model_inner(policy, lstm, bug_lstm, optimizer, bug_optimizer,
                                       env, bug_env, epochs=100, history_window=4, hoare_threshold=0.01,
                                       broken_training_epochs=5, correct_training_epoch=5,
                                       fresh_lstm=False, cuda=False, delta=False, action_input=False,
                                       batch_size=4, test_every=5, eps=0, no_correct_lstm_training=False,
                                       alpha=1, beta=1, max_t=50):

    test_accu = []
    bug_precision, bug_recall = [], []
    train_losses, bug_train_losses = [], []

    correct_traj_states, broken_traj_states = [], []

    for e in tqdm(range(epochs)):

        if not no_correct_lstm_training:
            states, actions, rewards = [], [], []
            state = env.reset()
            states.append(state)
            for i in range(max_t):
                # replace here with real policy
                action = policy.get_action(state, eps=eps)
                state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                if done:
                    break

            states = np.vstack(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            if action_input:
                lstm.store(states, actions, rewards)
            else:
                lstm.store(states)  # 1 data point in the batch
            correct_traj_states.append(states)

            if e % 5 == 0:
                if fresh_lstm:
                    lstm.reset()
                    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
                    train_epochs = 20
                else:
                    train_epochs = correct_training_epoch
                for _ in range(train_epochs):
                    loss = train_lstm_one_epoch(lstm, optimizer, l1_loss, cuda=cuda)
                    train_losses.append(np.mean(loss))  # average batch loss

        states, actions, rewards = [], [], []
        state = bug_env.reset()
        states.append(state)
        for i in range(max_t):
            # replace here with real policy
            action = policy.get_action(state, eps=eps)
            state, reward, done, _ = bug_env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                break

        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        if action_input:
            bug_lstm.store(states, actions, rewards)
        else:
            bug_lstm.store(states)

        broken_traj_states.append(states)

        if e % 5 == 0:
            if fresh_lstm:
                bug_lstm.reset()
                bug_optimizer = optim.Adam(bug_lstm.parameters(), lr=1e-3)
                train_epochs = 30
            else:
                train_epochs = broken_training_epochs

            for _ in range(train_epochs):
                loss = train_lstm_one_epoch(bug_lstm, bug_optimizer, l1_loss, cuda=cuda)
                bug_train_losses.append(np.mean(loss))  # average batch loss

        # get test set
        if e % test_every == 0:
            preds_y, test_y = [], []
            for _ in range(5):
                states, actions = [], []
                preds, labels = [], []

                state = bug_env.reset()
                hiddens, bug_hiddens = None, None

                # labels.append(0)
                for i in range(max_t):

                    # replace here with real policy
                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
                    bug_y_hat, bug_rew_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

                    # next-state
                    state, reward, done, info = bug_env.step(action)
                    if info['bug_state']:
                        labels.append(1)
                    else:
                        labels.append(0)

                    dist_to_correct = alpha * lstm.get_distance(y_hat, state)
                    dist_to_broken = alpha * bug_lstm.get_distance(bug_y_hat, state)

                    dist_to_correct += beta * lstm.get_reward_distance(rew_hat, reward)
                    dist_to_broken += beta * bug_lstm.get_reward_distance(bug_rew_hat, reward)

                    if np.abs(dist_to_correct - dist_to_broken) < hoare_threshold:
                        preds.append(0)  # 0 means correct
                    else:
                        preds.append(np.argmin([dist_to_correct, dist_to_broken]))

                    states.append(state)
                    actions.append(action)

                    if done:
                        break

                assert len(preds) == len(labels)
                states = np.vstack(states)
                preds_y.append(np.array(preds))
                test_y.append(np.array(labels))

            test_y = np.concatenate(test_y)
            preds_y = np.concatenate(preds_y)

            accu = accuracy_score(test_y, preds_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, preds_y)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (train_losses, bug_train_losses, lstm, bug_lstm,
                                                  correct_traj_states, broken_traj_states)

def collect_classifier_behavior_on_traj(agent, env, bug_env, lstm, bug_lstm, epochs,
                                        max_t=30, eps=0, threshold=0.1, cuda=False):

    traj_accus = []
    traj_dists = []  # each dist is a feature for classfication
    traj_labels, traj_preds = [], []

    for e in range(epochs):
        preds_y, test_y = [], []

        states = []
        preds, labels = [], []

        hiddens, bug_hiddens = None, None

        dists = []

        state = bug_env.reset()
        states.append(states)
        for i in range(max_t):
            # do a prediction here
            action = agent.get_action(state, eps=eps)

            inp = (state, action)
            y_hat, rew_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
            bug_y_hat, bug_rew_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

            # replace here with real policy
            state, reward, done, info = bug_env.step(action)  #
            if info['bug_state']:
                labels.append(1)
            else:
                labels.append(0)

            states.append(state)

            dist_to_correct = lstm.get_distance(y_hat, state)
            dist_to_broken = bug_lstm.get_distance(bug_y_hat, state)

            if np.abs(dist_to_correct - dist_to_broken) < threshold:
                preds.append(0)  # 0 means correct
            else:
                preds.append(np.argmin([dist_to_correct, dist_to_broken]))

            dists.append((dist_to_correct - dist_to_broken).numpy().squeeze())

            if done:
                break

        assert len(preds) == len(labels)

        preds_y.append(np.array(preds))
        test_y.append(np.array(labels))

        test_y = np.concatenate(test_y)
        preds_y = np.concatenate(preds_y)

        accu = accuracy_score(preds_y, test_y)
        traj_accus.append(accu)
        traj_dists.append(dists)
        traj_labels.append(test_y)
        traj_preds.append(preds_y)

    return traj_accus, traj_dists, traj_labels, traj_preds


def fix_single_quote_in_json(json_str):
    partial_fix = json_str.replace("'", '"')
    for a in ["\"random\"", "\"very slow\"", "\"slow\"", "\"normal\"", "\"fast\"", "\"very fast\"",
              '"hardcourt"', '"retro"']:
        b = a.replace('"', "'")
        partial_fix = partial_fix.replace(a, b)

    return json_str