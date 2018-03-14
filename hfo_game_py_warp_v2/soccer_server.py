import os
import subprocess
import time
import signal
import math

import logging
logger = logging.getLogger(__name__)

import hfo


class SoccerServer:
    def __init__(self):
        self.hfo_path = hfo.get_hfo_path()
        self.start_hfo_server()
        self.env = hfo.HFOEnvironment()
        self.server_process = None
        self.server_port = None
        self.env.connectToServer(config_dir=hfo.get_config_path())


    def start_hfo_server(self, frames_per_trial=500,
                          untouched_time=100, offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, sync_mode=True, port=6000,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """
        self.server_port = port
        # " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i" \
        cmd = self.hfo_path + \
              " --no-sync --frames-per-trial %i --untouched-time %i --offense-agents %i" \
              " --defense-agents %i --offense-npcs %i --defense-npcs %i" \
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f" \
              " --ball-x-max %f --log-dir %s" \
              % (frames_per_trial, untouched_time, offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10)  # Wait for server to startup before connecting a player

    def __del__(self):
        os.kill(self.server_process.pid, signal.SIGINT)
