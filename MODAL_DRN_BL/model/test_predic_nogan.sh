#!/bin/bash
nohup python -u predic_nogan.py test -a modal_drn_c_26 --resume modal_drn_c_26_model_nogan_best.pth.tar -g 1 > nogantestmodal_drn26pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_a_50 --resume modal_drn_a_50_model_nogan_best.pth.tar -g 1 > nogantestmodal_drn50pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_c_42 --resume modal_drn_c_42_model_nogan_best.pth.tar -g 1 > nogantestmodal_drn42pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_c_58 --resume modal_drn_c_58_model_nogan_best.pth.tar -g 2 > nogantestmodal_drn58pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_105 --resume modal_drn_d_105_model_nogan_best.pth.tar -g 2 > nogantestmodal_drn105pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_107 --resume modal_drn_d_107_model_nogan_best.pth.tar -g 2 > nogantestmodal_drn107pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_22  --resume modal_drn_d_22_model_nogan_best.pth.tar -g 3 > nogantestmodal_drn22pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_24  --resume modal_drn_d_24_model_nogan_best.pth.tar -g 3 > nogantestmodal_drn24pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_38  --resume modal_drn_d_38_model_nogan_best.pth.tar -g 3 > nogantestmodal_drn38pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_40  --resume modal_drn_d_40_model_nogan_best.pth.tar -g 1 > nogantestmodal_drn40pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_54  --resume modal_drn_d_54_model_nogan_best.pth.tar -g 2 > nogantestmodal_drn54pre.log 2>&1 &
nohup python -u predic_nogan.py test -a modal_drn_d_56  --resume modal_drn_d_56_model_nogan_best.pth.tar -g 3 > nogantestmodal_drn56pre.log 2>&1 &

