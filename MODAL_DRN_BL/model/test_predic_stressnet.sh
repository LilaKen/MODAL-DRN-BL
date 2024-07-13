#!/bin/bash
nohup python -u predic_stressnet.py test --resume ../output/stressnet/model_stressnet_best.pth.tar -g 1 > ../log/teststressnet_pre.log 2>&1 &
nohup python -u predic_scsnet.py test --resume ../output/scsnet/model_scsnet_best.pth.tar -g 1 > ../log/testscsnet_pre.log 2>&1 &
nohup python -u predic_inbetween.py  test --resume ../output/inbetween/model_inbetween_best.pth.tar -g 1 > ../log/testinbetween_pre.log 2>&1 &

