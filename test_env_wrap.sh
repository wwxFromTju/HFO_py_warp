#!/bin/bash

./bin/HFO --offense-agent=2 --trials 20 --no-sync & #  --defense-npcs=1
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 4
python test_env_warp.py --agent-name agent1 &
# python example/sarsa_offense/high_level_sarsa_agent.py &
sleep 4
python test_env_warp.py --agent-name agent2 &
sleep 4

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait