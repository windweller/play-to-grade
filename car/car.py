
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class CarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, mode='easy', bug=False):
        assert mode in {'easy', 'hard'}

        self.mode = mode
        self.bug = bug

        self.velocity = np.zeros(2)  # horizontal, vertical
        self.position = np.zeros(2)

        self.min_position = -10
        self.max_position = 10

        self.min_x = -10
        self.min_y = -10

        self.max_x = 10
        self.max_y = 10

        self.max_speed = 1
        self.force = 0.2

        self.viewer = None

        # left, right, up, down
        self.action_space = spaces.Discrete(4)

        self.low = np.array([self.min_position, self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_position, self.max_speed])

        self.seed()

        self.bug_triggered = False

        self.medium_high_end, self.medium_low_end, self.medium_easiness = self.sample_medium_start_end()


    def __repr__(self):
        if self.mode == 'easy':
            return f"CarEnv mode {self.mode}"
        elif self.mode == 'hard':
            return f"CarEnv mode {self.mode}, trigger: {self.medium_high_end, self.medium_low_end}, Easiness: {self.medium_easiness}"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_medium_start_end(self):
        # it's a square box, and make sure single direction agent fails
        # need to sample beyond [-2, 2]
        # [3, 9]
        # it's symmetrical, one start/end, we apply for all sides

        two_points = self.np_random.choice(range(3, 10), 2, replace=False)
        high_end = max(two_points)
        low_end = min(two_points)

        # the higher the overlap, the easier things get
        easiness = (high_end - low_end) / 10

        # high, low based on magnitude (absolute value)
        return high_end, low_end, easiness

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # boundary is a square box
        if self.bug and self.mode == 'easy':

            if self.bug_triggered is False:
                bug_boundary_max = 8
                bug_boundary_min = -8
                in_bug_area = bool(self.position[0] >= bug_boundary_max or self.position[0] <= bug_boundary_min or \
                                   self.position[1] >= bug_boundary_max or self.position[1] <= bug_boundary_min)
                if in_bug_area:
                    self.bug_triggered = True

            # once in, you won't get out...
            if self.bug_triggered:
                # action stopped being relevant, car just wiggles back and forth
                self.velocity = np.clip(self.velocity, -0.5, 0.5)
                self.velocity = -self.velocity  # flip velocity
                self.position += self.velocity

                done = False
                reward = -1

                # extra info helps us classify things
                return np.concatenate([self.position, self.velocity]), reward, done, {'bug_state': True}

        # boundary is a few line segments
        # play to win in this game is: driving out of the outer rim (however you choose to drive)
        if self.bug and self.mode == 'hard':
            # instead of a universal boundary (literally easy to get there)
            # it needs to go to a specific point, that's close to goal
            # where play-to-win can do fine
            # it literally is a box, with one side as 2, the other randomly sampled
            if self.bug_triggered is False:
                x, y = self.position[0], self.position[1]
                upper_left = x >= -self.medium_high_end and x <= -self.medium_low_end and y >= self.max_y - 2
                upper_right = x <= self.medium_high_end and x >= self.medium_low_end and y >= self.max_y - 2
                left_upper = y <= self.medium_high_end and y >= self.medium_low_end and x <= self.min_x + 2  # -8
                left_lower = y >= -self.medium_high_end and y <= -self.medium_low_end and x <= self.min_x + 2
                right_upper = y <= self.medium_high_end and y >= self.medium_low_end and x >= self.max_x - 2
                right_lower = y >= -self.medium_high_end and y <= -self.medium_low_end and x >= self.max_x - 2
                lower_right = x >= -self.medium_high_end and x <= -self.medium_low_end and y <= self.min_y + 2
                lower_left = x <= self.medium_high_end and x >= self.medium_low_end and y <= self.min_y + 2
                in_bug_area = upper_left or upper_right or left_upper or left_lower or right_upper or right_lower \
                              or lower_right or lower_left

                if in_bug_area:
                    self.bug_triggered = True

            # once in, you won't get out...
            if self.bug_triggered:
                # action stopped being relevant, car just wiggles back and forth
                self.velocity = np.clip(self.velocity, -0.5, 0.5)
                self.velocity = -self.velocity  # flip velocity
                self.position += self.velocity

                done = False
                reward = -1

                # extra info helps us classify things
                return np.concatenate([self.position, self.velocity]), reward, done, {'bug_state': True}

        # action is to apply acceleration/velocity change to each
        # currently just linear movement
        if action == 0:
            # left
            self.velocity += np.array([-self.force, 0])
        elif action == 1:
            # right
            self.velocity += np.array([self.force, 0])
        elif action == 2:
            # up
            self.velocity += np.array([0, self.force])
        elif action == 3:
            self.velocity += np.array([0, -self.force])

        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position += self.velocity

        if self.mode == 'easy' or self.mode == 'hard':
            # drive out of frame
            done = bool(self.position[0] >= self.max_position or self.position[0] <= self.min_position or \
                        self.position[1] >= self.max_position or self.position[1] <= self.min_position)

        self.position = np.clip(self.position, self.min_position, self.max_position)

        reward = -1  # same as mountain car

        return np.concatenate([self.position, self.velocity]), reward, done, {'bug_state': False}

    def reset(self):
        self.velocity = np.zeros(2)
        self.position = self.np_random.uniform(-2, 2, 2)
        self.bug_triggered = False

        return np.concatenate([self.position, self.velocity])