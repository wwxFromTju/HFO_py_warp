import math

import hfo

ACTION_LOOKUP = {
    0: hfo.DASH,
    1: hfo.TURN,
    2: hfo.KICK,
    3: hfo.TACKLE, # Used on defense to slide tackle the ball
    4: hfo.CATCH,  # Used only by goalie to catch the ball
}


class SoccerEnv():
    def __init__(self, env):
        self.env = env
        self.status = hfo.IN_GAME
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True

    def __del__(self):
        self.env.act(hfo.QUIT)
        self.env.step()

    def step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self.reward()
        ob = self.env.getState()
        episode_over = self.status != hfo.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo.NOOP)

    def reward(self):
        current_state = self.env.getState()
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity

        kickable = current_state[12]

        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.

        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.

        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)

        ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                                   2.*ball_dist*goal_dist*math.cos(alpha))

        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal

        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal

        reward = 0
        if not self.first_step:
            # Reward the agent for moving towards the ball
            reward += ball_prox_delta
            if kickable_delta > 0 and not self.got_kickable_reward:
                reward += 1.
                self.got_kickable_reward = True
            # Reward the agent for kicking towards the goal
            reward += 0.6 * -ball_dist_goal_delta
            # Reward the agent for scoring
            if self.status == hfo.GOAL:
                reward += 5.0

        self.first_step = False
        return reward

    def reset(self):
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True

        while self.status == hfo.IN_GAME:
            self.env.act(hfo.NOOP)
            self.status = self.env.step()
        while self.status != hfo.IN_GAME:
            self.env.act(hfo.NOOP)
            self.status = self.env.step()
        return self.env.getState()