#!/bin/bash
nohup python -u predic_nogan_bls.py train -a modal_drn_c_26 -g 0 > nogan_modal_drn_bls26pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_a_50 -g 0 > nogan_modal_drn_bls50pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_c_42 -g 0 > nogan_modal_drn_bls42pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_c_58 -g 2 > nogan_modal_drn_bls58pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_105 -g 2 > nogan_modal_drn_bls105pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_107 -g 2 > nogan_modal_drn_bls107pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_22 -g 3 > nogan_modal_drn_bls22pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_24 -g 3 > nogan_modal_drn_bls24pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_38 -g 3 > nogan_modal_drn_bls38pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_40 -g 0 > nogan_modal_drn_bls40pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_54 -g 2 > nogan_modal_drn_bls54pre.log 2>&1 &
nohup python -u predic_nogan_bls.py train -a modal_drn_d_56 -g 3 > nogan_modal_drn_bls56pre.log 2>&1 &
