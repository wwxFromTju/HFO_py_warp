import argparse

import numpy as np

from hfo import *

from hfo_game_py_warp import EnvWarp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-name', type=str)
    args = parser.parse_args()

    EnvWarp = EnvWarp(feature_set=LOW_LEVEL_FEATURE_SET,
                      config='bin/teams/base/config/formations-dt',
                      port=6000,
                      host='localhost',
                      side='base_left',
                      log=True,
                      agent_name=args.agent_name)

    for episode in range(100):
        status = IN_GAME

        state, reward, info = EnvWarp.go_to_ball()

        while status == IN_GAME:
            if state.kick_able == 1:
                state, reward, info = EnvWarp.shoot_to_goal()
            else:
                state, reward, info = EnvWarp.go_to_ball()

            print(reward)

        # use Q to test
        #
        # while status == IN_GAME:
        #     Q_value = np.random.uniform(-1, 1, 10)
        #     Q_value[0]  = Q_value[0] + 100
        #     # print(Q_value)
        #     for i in range(10):
        #         state, reward, info = EnvWarp.step(Q_value)
        #         print(reward)
        # status = info
        # if state[12] == 1:
        #     EnvWarp.shoot_to_goal()