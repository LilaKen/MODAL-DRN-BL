#!/bin/bash
nohup python -u predic_stressnet.py train  -g 1 > ../log/stressnet_pre.log 2>&1 &
nohup python -u predic_scsnet.py train  -g 1 > ../log/scsnet_pre.log 2>&1 &
nohup python -u predic_inbetween.py train  -g 1 > ../log/inbetween_pre.log 2>&1 &

