#!/bin/bash
nohup python -u predic_nogan.py train -a res_c_26 -g 2 > nogan_res26pre.log 2>&1 &
nohup python -u predic_nogan.py train -a res_c_42 -g 2 > nogan_res42pre.log 2>&1 &
nohup python -u predic_nogan.py train -a res_d_22 -g 1 > nogan_res22pre.log 2>&1 &
nohup python -u predic_nogan.py train -a res_d_38 -g 1 > nogan_res38pre.log 2>&1 &

