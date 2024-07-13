#!/bin/bash
nohup python -u predic_nogan.py test -a res_c_26 --resume res_c_26_model_nogan_resnet_best.pth.tar -g 1 > nogantestres26pre.log 2>&1 &
nohup python -u predic_nogan.py test -a res_c_42 --resume res_c_42_model_nogan_resnet_best.pth.tar -g 1 > nogantestres42pre.log 2>&1 &
nohup python -u predic_nogan.py test -a res_d_22  --resume res_d_22_model_nogan_resnet_best.pth.tar -g 2 > nogantestres22pre.log 2>&1 &
nohup python -u predic_nogan.py test -a res_d_38  --resume res_d_38_model_nogan_resnet_best.pth.tar -g 2 > nogantestres38pre.log 2>&1 &


#后期修改resume 添加nogan
