import numpy as np

import hfo

from hfo_game_py_warp_v2.soccer_env import SoccerEnv

feature_set = hfo.LOW_LEVEL_FEATURE_SET
config = '/Users/codeMan/Documents/hfo/HFO/bin/teams/base/config/formations-dt'
port = 6000
host = 'localhost'
side = 'base_left'


# ACTION_LOOKUP = {
#     0: hfo.DASH,
#     1: hfo.TURN,
#     2: hfo.KICK,
#     3: hfo.TACKLE, # Used on defense to slide tackle the ball
#     4: hfo.CATCH,  # Used only by goalie to catch the ball
# }

server = hfo.HFOEnvironment()
server.connectToServer(feature_set, config, port, host, side, False)

env = SoccerEnv(env=server)


for i in range(1000):
    rand_action_index = np.random.randint(0, 3)

    rand_dash_pow = np.random.uniform(0, 100)
    rand_dash_dic = np.random.uniform(-180, 180)

    rand_turn_dic = np.random.uniform(-180, 180)

    rand_kick_pow = np.random.uniform(0, 100)
    rand_kick_dic = np.random.uniform(-180, 180)

    s, r, d, _ = env.step([rand_action_index,
                  rand_dash_pow, rand_dash_dic,
                  rand_turn_dic,
                  rand_kick_pow, rand_kick_dic])

    print(s, r, d, _)

