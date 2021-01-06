from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension,CppExtension,CUDAExtension
import os
from os.path import join as pjoin

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_HOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

setup(
    name='torch_nndistance',
    ext_modules=[
        CUDAExtension(
            'torch_nndistance_aten',
            ['capsulenet/torch-nndistance/src/my_lib_cuda.cpp', 'capsulenet/torch-nndistance/src/nnd_cuda.cu']
        )],
    cmdclass={'build_ext': BuildExtension},
    #packages=["torch_nndistance"],
    #classifiers=[
    #    'Programming Language :: Python :: 3',
    #],
)


__version__ = '1.0.0'

install_requires = [
    'torch>=1.0.0',
    'numpy',
    'tqdm',
    'scipy',
    'networkx',
    'scikit-learn',
    'sklearn',
    'requests',
    'h5py'
]

setup(name='my_lib_cuda',
      ext_modules=[CUDAExtension('my_lib_cuda',['capsulenet/nndistance/src/my_lib_cuda.cpp', 'capsulenet/nndistance/src/nnd_cuda.cu']
              )],
      cmdclass={'build_ext': BuildExtension}
      )

setup(
    name='capsulenet',
    version=__version__,
    description='3D Capsule Network in Pytorch',
    author='Gayathri Mahalingam',
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True
)