import os
import torch
import torch.utils.ffi

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'
strHeaders = []
strSources = []
strDefines = []
strObjects = []

if torch.cuda.is_available() == True:
    strHeaders += ['src/gpu18.h']
    strSources += ['src/gpu18.c']
    strDefines += [('WITH_CUDA', None)]
    strObjects += ['src/RisiContraction_18_gpu_cuda.o']
# end

print("Start1")
objectExtension = torch.utils.ffi.create_extension(
    name='_ext.gpu18',
    headers=strHeaders,
    sources=strSources,
    verbose=True,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
)

print("Start2")
if __name__ == '__main__':
    objectExtension.build()

print("Start3")
# end
