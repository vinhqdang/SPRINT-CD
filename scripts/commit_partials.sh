#!/bin/sh
# Commit and push checkpoint files of long experiments so a container restart does not lose them.
cd /home/user/SPRINT-CD || exit 1
git add results/*.partial.json 2>/dev/null
git -c user.name=Vinh -c user.email=dqvinh87@gmail.com commit -q -m "Checkpoint of running experiments" 2>/dev/null && git push -q origin HEAD:main 2>/dev/null
