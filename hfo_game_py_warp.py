import numpy as np

from hfo import *

k_pass_vel_threshold = -.5

class Feature:
    def __init__(self, features, server):
        self.features = features
        self.ball_proximity = features[53]
        self.goal_proximity = features[15]

        self.ball_dist = 1.0 - self.ball_proximity
        self.goal_dist = 1.0 - self.goal_proximity

        self.kick_able = features[12]

        self.ball_ang_sin_rad = features[51]
        self.ball_ang_cos_rad = features[52]

        # 注意一下角度之类的
        ball_ang_rad = np.arccos(self.ball_ang_cos_rad)
        self.ball_ang_rad = ball_ang_rad * -1 if self.ball_ang_sin_rad < 0 else ball_ang_rad

        self.gola_ang_sin_rad = features[13]
        self.gola_ang_cos_rad = features[14]

        # 注意一下弧度和角度的关系
        goal_ang_rad = np.arccos(self.gola_ang_cos_rad)
        self.goal_ang_rad = goal_ang_rad * -1 if self.gola_ang_sin_rad < 0 else goal_ang_rad

        self.alpha = np.max([self.ball_ang_rad, goal_ang_rad]) - np.min([self.ball_ang_rad, self.goal_ang_rad])

        self.ball_dist_goal = np.sqrt(self.ball_dist ** 2 + self.goal_dist ** 2 - 2.0 * self.ball_dist * self.goal_dist * np.cos(self.alpha))

        self.ball_vel_vaild = features[54]
        self.ball_vel = features[55]

        self.player_on_ball = server.playerOnBall()

        self.player_on_ball_unum = self.player_on_ball.unum

        self.agent_unum = server.getUnum()


class EnvWarp:
    def __init__(self, feature_set, config, port, host, side, log=False, agent_name=''):
        self.server = HFOEnvironment()
        self.server.connectToServer(feature_set, config, port, host, side, False)

        self.old_state = None
        self.now_state = None
        self.info = None

        self.log = log

        self.got_kickable_reward = False

        self.pass_active = False

        self.step_num = 0
        self.agent_name = agent_name

    def shoot_to_goal(self):
        self.server.act(SHOOT)

        self.info = self.server.step()

        state = Feature(self.server.getState(), self.server)
        self.old_state = self.now_state
        self.now_state = state

        reward = self.reward()

        if self.info == GOAL:
            self.old_state = None

        return state, reward, self.info

    def go_to_ball(self):
        self.server.act(GO_TO_BALL)

        self.info = self.server.step()

        state = Feature(self.server.getState(), self.server)
        self.old_state = self.now_state
        self.now_state = state

        reward = self.reward()

        if self.info == GOAL:
            self.old_state = None

        return state, reward, self.info

    def add_step(self):
        def warp():
            self.step_num += 1

        return warp

    @add_step
    def step(self, Q):
        dash_q, turn_q, tackle_q, kick_q = Q[0:4]

        dash_pow, dash_dic = Q[4:6]
        turn_dic = Q[6]
        tackle_dic = Q[7]
        kick_pow, kick_dic = Q[8:]

        action_index = np.random.choice(np.where(np.array(Q[0:4]) == np.max(Q[0:4]))[0])

        if action_index == 0:
            self.dash(dash_pow, dash_dic)
        elif action_index == 1:
            self.turn(turn_dic)
        elif action_index == 2:
            self.tackle(tackle_dic)
        elif action_index == 3:
            self.kick(kick_pow, kick_dic)
        else:
            print('error action')

        self.info = self.server.step()

        state = Feature(self.server.getState(), self.server)
        self.old_state = self.now_state
        self.now_state = state

        reward = self.reward()

        if self.info == GOAL:
            self.old_state = None

        if self.now_state.ball_vel_vaild and self.now_state.ball_vel > k_pass_vel_threshold:
            self.pass_active = True

        return state, reward, self.info

    def reward(self):
        if self.old_state is None:
            return 0
        return self.cal_reward()

    def move_to_ball_reward(self):
        ball_prox_delta = self.now_state.ball_proximity - self.old_state.ball_proximity
        kick_able_delta = self.now_state.kick_able - self.old_state.kick_able
        move_to_ball_reward = 0
        if self.now_state.player_on_ball_unum < 0 or self.now_state.agent_unum == self.now_state.player_on_ball_unum:
            move_to_ball_reward += ball_prox_delta
        self.got_kickable_reward = False
        if kick_able_delta >= 1 and not self.got_kickable_reward  and self.now_state.player_on_ball_unum == self.now_state.agent_unum:
            move_to_ball_reward += 1.0
            self.got_kickable_reward = True

        return move_to_ball_reward

    def kick_to_goal_reward(self):
        ball_dist_goal_delta = self.now_state.ball_dist_goal - self.old_state.ball_dist_goal
        kick_to_goal_reward = 0
        if self.now_state.player_on_ball_unum == self.now_state.agent_unum:
            if self.log:
                print('{} kick ball reward, ball unum: {}'.format(self.agent_name, self.now_state.player_on_ball_unum), 'agent unu, {}'.format(self.now_state.agent_unum))
            kick_to_goal_reward = - ball_dist_goal_delta
        elif self.got_kickable_reward:
            if self.log:
                print('team kick ball reward')
            kick_to_goal_reward = 0.2 * - ball_dist_goal_delta

        return kick_to_goal_reward

    def pass_reward(self):
        pass_reward = 0

        if self.pass_active and self.now_state.player_on_ball_unum > 0 and \
            self.now_state.player_on_ball_unum != self.old_state.player_on_ball_unum:
            self.pass_active = False

            if self.log:
                print("{}, Unum: {}, steps: {}  got pass reward!".format(self.agent_name, self.now_state.agent_unum, self.step_num))

            pass_reward += 1

        return pass_reward

    def eot_reward(self):
        eot_reward = 0
        if self.info == GOAL:
            if self.old_state.player_on_ball.side == RIGHT:
                if self.log:
                    print("{} Unexpected side: {}".format(self.agent_name, self.old_state.player_on_ball.side))
                return eot_reward
            if self.old_state.player_on_ball_unum == self.now_state.agent_unum:
                if self.log:
                    print('{} scored!, ball unum: {}'.format(self.agent_name, self.now_state.player_on_ball_unum), 'agent unu, {}'.format(self.now_state.agent_unum))
                eot_reward += 5
            else:
                if self.log:
                    print('team scored!', 'ball unum: {}'.format(self.now_state.player_on_ball_unum), 'agent unu, {}'.format(self.now_state.agent_unum))
                eot_reward += 1
        elif self.info == CAPTURED_BY_DEFENSE:
            pass

        return eot_reward

    def cal_reward(self):
        # ball_prox_delta = self.now_state.ball_proximity - self.old_state.ball_proximity
        # kick_able_delta = self.now_state.kick_able - self.old_state.kick_able
        # ball_dist_goal_delta = self.now_state.ball_dist_goal - self.old_state.ball_dist_goal
        #
        # # move to ball reward
        # move_to_ball_reward = 0
        # if self.now_state.player_on_ball_unum < 0 or self.now_state.agent_unum == self.now_state.player_on_ball_unum:
        #     move_to_ball_reward += ball_prox_delta
        # got_kickable_reward = False
        # if kick_able_delta >= 1 and not got_kickable_reward:
        #     move_to_ball_reward += 1.0
        #     got_kickable_reward = True
        #
        # # kick to goal reward
        # kick_to_goal_reward = 0
        # if self.now_state.player_on_ball_unum == self.now_state.agent_unum:
        #     kick_to_goal_reward = - ball_dist_goal_delta
        # elif got_kickable_reward:
        #     kick_to_goal_reward = 0.2 * - ball_dist_goal_delta
        #
        # # passing reward
        # pass_reward = 0
        #
        # # EOT reward
        # eot_reward = 0

        totol_reward = self.move_to_ball_reward() + 3 * self.kick_to_goal_reward() + 3 * self.pass_reward() + \
                       self.eot_reward()

        return totol_reward

    def dash(self, power, dic):
        '''
        power in [-1, 1]
        dic in [-1 , 1]
        '''
        if self.log:
            print('dash, power: {}, dic: {}'.format(power, dic))
        assert -1 <= power <= 1, 'power must in [-1, 1]'
        assert -1 <= dic <= 1, 'dic must in [-1, 1]'
        power *= 100
        dic *= 180
        self.server.act(DASH, power, dic)

    def tackle(self, dic):
        '''
        dic in [-1, 1]
        '''
        if self.log:
            print('tackle, dic: {}'.format(dic))
        assert -1 <= dic <= 1, 'dic must in [-1, 1]'
        dic *= 180
        self.server.act(TACKLE, dic)

    def turn(self, dic):
        '''
        dic in [-1, 1]
        '''
        if self.log:
            print('turn, dic: {}'.format(dic))
        assert -1 <= dic <= 1, 'dic must in [-1, 1]'
        dic *= 180
        self.server.act(TURN, dic)

    def kick(self, power, dic):
        '''
        power in [0, 1]
        dic in [-1, 1]
        '''
        if self.log:
            print('kick, power: {}, dic: {}'.format(power, dic))
        assert -1 <= power <= 1, 'power must in [-1, 1]'
        assert -1 <= dic <= 1, 'dic must in [-1, 1]'
        power = (power + 1) * 100
        dic *= 180
        self.server.act(KICK, power, dic)
