#!/bin/bash

/Users/codeMan/Documents/hfo/HFO/bin/HFO --offense-agent=2 --trials 20 --no-sync & #  --defense-npcs=1

sleep 4
python test_env_warp_v2.py --agent-name agent1 &

sleep 4
python test_env_warp_v2.py --agent-name agent2 &


trap "kill -TERM -$$" SIGINT
wait