#!/bin/bash
nohup python -u predic.py test -a modal_drn_c_26 --resume modal_drn_c_26_model_best.pth.tar -g 1 > testmodal_drn26pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_a_50 --resume modal_drn_a_50_model_best.pth.tar -g 1 > testmodal_drn50pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_c_42 --resume modal_drn_c_42_model_best.pth.tar -g 1 > testmodal_drn42pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_c_58 --resume modal_drn_c_58_model_best.pth.tar -g 2 > testmodal_drn58pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_105 --resume modal_drn_d_105_model_best.pth.tar -g 2 > testmodal_drn105pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_107 --resume modal_drn_d_107_model_best.pth.tar -g 2 > testmodal_drn107pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_22  --resume modal_drn_d_22_model_best.pth.tar -g 3 > testmodal_drn22pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_24  --resume modal_drn_d_24_model_best.pth.tar -g 3 > testdmodal_rn24pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_38  --resume modal_drn_d_38_model_best.pth.tar -g 3 > testmodal_drn38pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_40  --resume modal_drn_d_40_model_best.pth.tar -g 1 > testmodal_drn40pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_54  --resume modal_drn_d_54_model_best.pth.tar -g 2 > testmodal_drn54pre.log 2>&1 &
nohup python -u predic.py test -a modal_drn_d_56  --resume modal_drn_d_56_model_best.pth.tar -g 3 > testmodal_drn56pre.log 2>&1 &

#后期修改resume 添加nogan