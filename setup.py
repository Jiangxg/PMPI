# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

# build aabb
_ext_src_root = "aabb"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# what is the meaning of each parameter?
_ext_headers_path = os.getcwd() + "/" + _ext_src_root + "/include"

setup(
    name='aabb',
    ext_modules=[
        CUDAExtension(
            name='aabb._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format(_ext_headers_path)],
                "nvcc": ["-O2", "-I{}".format(_ext_headers_path)],
            },
            extra_link_args = ["-fopenmp"],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
