#!/bin/bash

#OAR -n gpu_go_brrrrr
#OAR -l /nodes=1,walltime=24:0:0
#OAR -q production

pixi shell
python -V
which python
pwd

# python -c "import torch; print(torch.cuda.is_available())"

$@