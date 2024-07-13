# Project README

This project contains various scripts and models for training and testing on a specific dataset. Below is a brief description of the directory structure and the purpose of each script.

## Directory Structure

- `dataset/`: The folder where the dataset is stored. To obtain the dataset, please contact the author via email at [lil_ken@163.com](mailto:lil_ken@163.com).
- `main/`: Contains the main training and testing scripts for different models and experiments.
- `models/`: Contains the model architectures used in the experiments.
- `utils/`: Contains utility scripts for data processing and other helper functions.

## Scripts

### Training Scripts

- `run_predic.sh`: Training script for GAN models.
- `run_predic_nogan.sh`: Training script for MODAL-DRN models.
- `run_predic_nogan_bls.sh`: Training script for MODAL-DRN-BL models.
- `run_predic_stressnet.sh`: Training script for StressNet, SCSNet, and Inbetween models.
- `run_predic_nogan_resnet.sh`: Training script for ablation experiments.

### Testing Scripts

- `test_predic.sh`: Testing script for GAN models.
- `test_predic_bls.sh`: Testing script for MODAL-DRN-BL models.
- `test_predic_nogan.sh`: Testing script for MODAL-DRN models.
- `test_predic_nogan_bls.sh`: Testing script for MODAL-DRN-BL models.
- `test_predic_nogan_resnet.sh`: Testing script for ablation experiments.

## Contact

If you have any questions about the code files, please contact the author at [lil_ken@163.com](mailto:lil_ken@163.com).

**PS:** This is the first public code experiment, and there may be many places that are not considerate enough. We appreciate your understanding.