# Training methods are stored here

from tqdm import tqdm

import numpy as np
import torch

from collections import deque
import torch.optim as optim

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from car.classifiers import VAE, train_vae_one_epoch, compute_vae_loss, HoareLSTM, train_lstm_one_epoch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from collections import deque

# ========= MLP =========

def train_naive_supervised_classifier(policy, env, bug_env, epochs=100, history_window=4):
    X = deque(maxlen=100)
    y = deque(maxlen=100)

    feat_dim = 4

    cls = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1,
                        max_iter=5)

    test_accu = []
    bug_precision, bug_recall = [], []

    # reset after 5 epochs
    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)
        y.append(np.zeros(states.shape[0]))

        # bug_env
        states = []
        state_buffer = deque(maxlen=history_window)

        state = bug_env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = bug_env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)
        y.append(np.ones(states.shape[0]))

        cls.fit(np.vstack(X), np.concatenate(y))

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                for i in range(30):
                    # replace here with real policy
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_y_hat = cls.predict(test_X)

            accu = accuracy_score(test_y_hat, test_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall

def train_naive_supervised_classifier_inner(policy, model, env, bug_env, epochs=100, history_window=4):
    X = deque(maxlen=100)
    y = deque(maxlen=100)

    feat_dim = 4
    test_accu = []
    bug_precision, bug_recall = [], []

    # reset after 5 epochs
    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)
        y.append(np.zeros(states.shape[0]))

        # bug_env
        states = []
        state_buffer = deque(maxlen=history_window)

        state = bug_env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = bug_env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)
        y.append(np.ones(states.shape[0]))

        model.fit(np.vstack(X), np.concatenate(y))

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                # labels.append(0)
                for i in range(30):
                    # replace here with real policy
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_y_hat = model.predict(test_X)

            accu = accuracy_score(test_y_hat, test_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall

# ========= GMM =============

def train_naive_gmm_model(policy, env, bug_env, epochs=100, history_window=4):
    X = deque(maxlen=100)

    feat_dim = 4

    model = GaussianMixture(n_components=5, covariance_type='full', max_iter=500)

    normalizer = Normalizer(norm='l1')

    test_accu = []
    bug_precision, bug_recall = [], []
    train_losses = []

    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)

        normalizer = normalizer.fit(np.vstack(X))
        model.fit(normalizer.transform(np.vstack(X)))

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                # labels.append(0)
                for i in range(30):
                    # replace here with real policy
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_X = normalizer.transform(test_X)
            test_y_hat = model.predict(test_X)

            accu = accuracy_score(test_y_hat, test_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, train_losses

def train_naive_gmm_model_inner(policy, model,
                                env, bug_env,
                                normalizer=None, epochs=100, history_window=4):
    X = deque(maxlen=100)

    feat_dim = 4

    if normalizer is None:
        normalizer = Normalizer(norm='l1')

    test_accu = []
    bug_precision, bug_recall = [], []
    train_losses = []

    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)

        normalizer = normalizer.fit(np.vstack(X))
        model.fit(normalizer.transform(np.vstack(X)))

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                # labels.append(0)
                for i in range(30):
                    # replace here with real policy
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_X = normalizer.transform(test_X)
            test_y_hat = model.predict(test_X)

            accu = accuracy_score(test_y_hat, test_y)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (model, normalizer, train_losses)

# ===== VAE =======

def train_naive_vae_model(policy, env, bug_env, epochs=100, history_window=4, cuda=False):
    X = deque(maxlen=100)

    feat_dim = 4

    model = VAE(feat_dim * history_window)
    if cuda:
        model = model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    test_accu = []
    bug_precision, bug_recall = [], []

    train_losses = []

    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            # replace here with real policy
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)

        # inner VAE optimization loop
        for _ in range(10):
            loss = train_vae_one_epoch(model, optimizer, torch.from_numpy(np.vstack(X)).float(), cuda=cuda)
            train_losses.append(loss)

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                # labels.append(0)
                for i in range(30):
                    # replace here with real policy
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_y_hat = compute_vae_loss(model, torch.from_numpy(np.vstack(test_X)).float(), cuda=cuda) > np.mean(train_losses[-100:])

            accu = accuracy_score(test_y, test_y_hat)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (model, train_losses)

def train_naive_vae_model_inner(policy, model, optimizer, env, bug_env, epochs=100, history_window=4, cuda=False):
    test_accu = []
    bug_precision, bug_recall = [], []

    train_losses = []

    X = deque(maxlen=100)

    for e in tqdm(range(epochs)):
        states = []
        state_buffer = deque(maxlen=history_window)

        state = env.reset()
        state_buffer.append(state)
        for i in range(30):
            state, reward, done, _ = env.step(policy.get_action(state))
            if len(state_buffer) == history_window:
                states.append(np.concatenate(state_buffer))
            state_buffer.append(state)
            if done:
                break
        states = np.vstack(states)
        X.append(states)

        # inner VAE optimization loop
        for _ in range(10):
            loss = train_vae_one_epoch(model, optimizer, torch.from_numpy(np.vstack(X)).float(), cuda=cuda)
            train_losses.append(loss)

        # get test set
        if e % 5 == 0:
            test_X, test_y = [], []
            for _ in range(10):
                states = []
                labels = []

                state = bug_env.reset()
                state_buffer = deque(maxlen=history_window)

                state_buffer.append(state)
                for i in range(30):
                    state, reward, done, info = bug_env.step(policy.get_action(state))
                    if len(state_buffer) == history_window:
                        states.append(np.concatenate(state_buffer))
                        if info['bug_state']:
                            labels.append(1)
                        else:
                            labels.append(0)

                    state_buffer.append(state)

                    if done:
                        break

                assert len(states) == len(labels)
                states = np.vstack(states)
                test_X.append(states)
                test_y.append(np.array(labels))

            test_X, test_y = np.vstack(test_X), np.concatenate(test_y)
            test_y_hat = compute_vae_loss(model, torch.from_numpy(np.vstack(test_X)).float(), cuda=cuda) > np.mean(train_losses[-100:])

            accu = accuracy_score(test_y, test_y_hat)
            test_accu.append(accu)

            prec, rec, f1, _ = precision_recall_fscore_support(test_y, test_y_hat)
            if np.sum(test_y) > 1:
                bug_precision.append(prec[1])  # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (model, train_losses)

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

                    # replace here with real policy
                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

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
            # sometimes there's just no bug at all
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
                                       batch_size=4, test_every=5, eps=0):
    test_accu = []
    bug_precision, bug_recall = [], []  # when we say it's a bug, it is indeed a bug
    train_losses, bug_train_losses = [], []

    correct_traj_states, broken_traj_states = [], []

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
        correct_traj_states.append(states)

        if e % 5 == 0:
            train_epochs = correct_training_epoch
            for _ in range(train_epochs):
                loss = train_lstm_one_epoch(lstm, optimizer, l1_loss, cuda=cuda)
                # print("Correct", np.mean(loss))
                train_losses.append(np.mean(loss))  # average batch loss

        # get test set
        if e % test_every == 0:
            preds_y, test_y = [], []
            for _ in range(10):
                states, actions = [], []
                preds, labels = [], []

                state = bug_env.reset()
                hiddens, bug_hiddens = None, None

                # labels.append(0)
                for i in range(30):

                    # replace here with real policy
                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)

                    # next-state
                    state, reward, done, info = bug_env.step(action)
                    if info['bug_state']:
                        labels.append(1)
                    else:
                        labels.append(0)

                    dist_to_correct = lstm.get_distance(y_hat, state)
                    # np.mean(train_losses[-10:]) +
                    preds.append(dist_to_correct > threshold_offset)

                    # preds.append(dist > np.mean(train_losses[-10:]))
                    states.append(state)
                    actions.append(action)

                    if done:
                        break

                assert len(preds) == len(labels)
                # states = np.vstack(states)
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
                                       env, bug_env, epochs, hoare_threshold=hoare_threshold,
                                       cuda=cuda, delta=delta, action_input=action_input, batch_size=batch_size)

# this is to help joint training with DQN
def train_naive_hoare_lstm_model_inner(policy, lstm, bug_lstm, optimizer, bug_optimizer,
                                       env, bug_env, epochs=100, history_window=4, hoare_threshold=0.01,
                                       broken_training_epochs=5, correct_training_epoch=5,
                                       fresh_lstm=False, cuda=False, delta=False, action_input=False,
                                       batch_size=4, test_every=5, eps=0, no_correct_lstm_training=False):
    # eps: we do want the policy to be relatively random; because we want to collect as much info as possible

    # dist1, dist2
    # dist1 - dist2 < epsilon_threshold
    # we call it correct, otherwise we go with whichever closer

    test_accu = []
    bug_precision, bug_recall = [], []  # when we say it's a bug, it is indeed a bug
    train_losses, bug_train_losses = [], []

    correct_traj_states, broken_traj_states = [], []

    # reset after 5 epochs
    for e in tqdm(range(epochs)):

        # we can train correctLSTM on the outside!! elsewhere, till FULL convergence
        # no need to keep it in here...
        # but keep it in here makes training somewhat balanced
        if not no_correct_lstm_training:
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

        states, actions = [], []
        state = bug_env.reset()
        states.append(state)
        for i in range(30):
            action = policy.get_action(state, eps=eps)
            state, reward, done, _ = bug_env.step(action)
            states.append(state)
            actions.append(action)
            if done:
                break

        states = np.vstack(states)
        actions = np.array(actions)
        if action_input:
            bug_lstm.store(states, actions)
        else:
            bug_lstm.store(states)  # 1 data point in the batch

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
            for _ in range(10):
                states, actions = [], []
                preds, labels = [], []

                state = bug_env.reset()
                hiddens, bug_hiddens = None, None

                for i in range(30):

                    # replace here with real policy
                    action = policy.get_action(state, eps=eps)

                    # do a prediction here
                    inp = state if not action_input else (state, action)
                    y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=cuda)
                    bug_y_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=cuda)

                    # next-state
                    state, reward, done, info = bug_env.step(action)
                    if info['bug_state']:
                        labels.append(1)
                    else:
                        labels.append(0)

                    dist_to_correct = lstm.get_distance(y_hat, state)
                    dist_to_broken = bug_lstm.get_distance(bug_y_hat, state)

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
                bug_precision.append(prec[1]) # 2nd label is bug
                bug_recall.append(rec[1])
            else:
                bug_precision.append(np.nan)
                bug_recall.append(np.nan)

    return test_accu, bug_precision, bug_recall, (train_losses, bug_train_losses, lstm, bug_lstm,
                                                  correct_traj_states, broken_traj_states)

def collect_trajs(agent, bug_env, epochs, max_t=30, eps=0):
    # reset after 5 epochs
    traj_states = []
    traj_labels = []
    for e in tqdm(range(epochs)):
        states, actions = [], []
        labels = []
        state = bug_env.reset()
        states.append(state)

        for i in range(max_t):
            # do a prediction here

            state, reward, done, info = bug_env.step(agent.get_action(state, eps))
            if info['bug_state']:
                labels.append(1)
            else:
                labels.append(0)

            states.append(state)

            if done:
                break

        states = np.vstack(states)
        traj_states.append(states)
        traj_labels.append(labels)

    return traj_states, traj_labels

def collect_classifier_behavior_on_traj(agent, env, bug_env, lstm, bug_lstm, epochs,
                                        max_t=30, eps=0, threshold=0.1):

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
            y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=True)
            bug_y_hat, bug_hiddens = bug_lstm.predict_post_state(inp, pre_state=bug_hiddens, cuda=True)

            state, reward, done, info = bug_env.step(action)  #
            if info['bug_state']:
                labels.append(1)
            else:
                labels.append(0)

            states.append(state)

            dist_to_correct = lstm.get_distance(y_hat, state)
            dist_to_broken = bug_lstm.get_distance(bug_y_hat, state)  # SAE, we'll put lstm here, instead of bug_lstm

            if np.abs(dist_to_correct - dist_to_broken) < threshold:
                preds.append(0)  # 0 means correct
            else:
                preds.append(np.argmin([dist_to_correct, dist_to_broken]))

            # decision boundary
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

def collect_sae_classifier_behavior_on_traj(agent, env, bug_env, lstm, epochs,
                                        max_t=30, eps=0, threshold=0.1):

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
            y_hat, hiddens = lstm.predict_post_state(inp, pre_state=hiddens, cuda=True)

            # replace here with real policy
            state, reward, done, info = bug_env.step(action)  #
            if info['bug_state']:
                labels.append(1)
            else:
                labels.append(0)

            states.append(state)

            dist_to_correct = lstm.get_distance(y_hat, state)
            preds.append(dist_to_correct > threshold)

            dists.append((dist_to_correct).numpy().squeeze())

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
