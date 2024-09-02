import glob
import os
import platform
import torch
from torch.utils.cpp_extension import BuildExtension
from pkg_resources import parse_version
from setuptools import find_packages, setup

EXT_TYPE = 'pytorch'
cmd_class = {'build_ext': BuildExtension}

install_requires = []


def get_extensions():
    extensions = []

    if os.getenv('MMCV_WITH_OPS', '1') == '0':
        return extensions

    if EXT_TYPE == 'pytorch':
        ext_name = 'mmcv_ops._ext'
        from torch.utils.cpp_extension import CUDAExtension

        # prevent ninja from using too many resources
        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []

        # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
        # required key passed to PyTorch. Even if there is no flag passed
        # to cxx, users also need to pass an empty list to PyTorch.
        # Since PyTorch1.8.0, it has a default value so users do not need
        # to pass an empty list anymore.
        # More details at https://github.com/pytorch/pytorch/pull/45956
        extra_compile_args = {'cxx': []}

        if platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] += ['-std=c++14']
            else:
                extra_compile_args['cxx'] += ['-std=c++17']
        else:
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] += ['/std:c++14']
            else:
                extra_compile_args['cxx'] += ['/std:c++17']

        include_dirs = []
        library_dirs = []
        libraries = []

        extra_objects = []
        extra_link_args = []
        is_rocm_pytorch = False
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                       (ROCM_HOME is not None)) else False
        except ImportError:
            pass

        if is_rocm_pytorch or torch.cuda.is_available() or os.getenv(
                'FORCE_CUDA', '0') == '1':
            if is_rocm_pytorch:
                define_macros += [('MMCV_WITH_HIP', None)]
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./**/*.cpp', recursive=True) + \
                glob.glob('./**/*.cu', recursive=True)
            extension = CUDAExtension
            include_dirs.append(os.path.abspath("./include"))
            for op in os.listdir('mmcv_ops'):
                op_path = os.path.join('mmcv_ops', op)
                if os.path.isdir(op_path):
                    op_include_dir = os.path.join(op_path, 'include')
                    include_dirs.append(os.path.abspath(op_include_dir))

        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']
        ext_ops = extension(name=ext_name,
                            sources=op_files,
                            include_dirs=include_dirs,
                            define_macros=define_macros,
                            extra_objects=extra_objects,
                            extra_compile_args=extra_compile_args,
                            library_dirs=library_dirs,
                            libraries=libraries,
                            extra_link_args=extra_link_args)
        extensions.append(ext_ops)
    return extensions


setup(name='mmcv_ops',
      version='1.0',
      description='OpenMMLab Computer Vision Foundation',
      keywords='computer vision',
      packages=find_packages(),
      include_package_data=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Utilities',
      ],
      url='https://github.com/LOTEAT/mmcv-ops',
      author='MMCV Contributors',
      author_email='zenglezhu@gmail.com',
      install_requires=install_requires,
      python_requires='>=3.7',
      ext_modules=get_extensions(),
      cmdclass=cmd_class,
      zip_safe=False)
