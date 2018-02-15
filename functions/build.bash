#!/bin/bash

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
rm -rf ../../_ext
rm -rf ../../functions/__pycache__
rm -rf src/RisiContraction_18_gpu_cuda.o

nvcc -c -o src/RisiContraction_18_gpu_cuda.o src/RisiContraction_18_gpu_cuda.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python build.py


cp -rf ./_ext ../../_ext

