#!/bin/bash
nohup python -u predic_nogan.py train -a modal_drn_c_26 -g 1 > nogan_modal_drn26pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_a_50 -g 1 > nogan_modal_drn50pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_c_42 -g 1 > nogan_modal_drn42pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_c_58 -g 2 > nogan_modal_drn58pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_105 -g 2 > nogan_modal_drn105pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_107 -g 2 > nogan_modal_drn107pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_22 -g 3 > nogan_modal_drn22pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_24 -g 3 > nogan_modal_drn24pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_38 -g 3 > nogan_modal_drn38pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_40 -g 1 > nogan_modal_drn40pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_54 -g 2 > nogan_modal_drn54pre.log 2>&1 &
nohup python -u predic_nogan.py train -a modal_drn_d_56 -g 3 > nogan_modal_drn56pre.log 2>&1 &
