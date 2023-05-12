#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""User friendly / multi platform builder script"""

import argparse
import datetime
import glob
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
import requests
from distutils.dir_util import copy_tree
from enum import Enum
from subprocess import check_call, call

from util import *

KNOWN_VCVARS = {
    # https://gitlab.kitware.com/cmake/cmake/issues/18920
    'VS 2015': r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat',
    'VS 2017': r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsx86_amd64.bat',
    'VS 2019': r'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat',
}


class BuildFlavour(Enum):
    WIN_CPU = 'WIN_CPU'
    WIN_CPU_ONEDNN = 'WIN_CPU_ONEDNN'
    WIN_CPU_ONEDNN_MKL = 'WIN_CPU_ONEDNN_MKL'
    WIN_CPU_MKL = 'WIN_CPU_MKL'
    WIN_GPU = 'WIN_GPU'
    WIN_GPU_ONEDNN = 'WIN_GPU_ONEDNN'


CMAKE_FLAGS = {
    'WIN_CPU': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=OFF '
        '-DUSE_CUDNN=OFF '
        '-DUSE_OPENCV=ON '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=open '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DBUILD_CPP_EXAMPLES=ON '
        '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_ONEDNN': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=OFF '
        '-DUSE_CUDNN=OFF '
        '-DUSE_OPENCV=ON '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=open '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DUSE_ONEDNN=ON '
        '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_ONEDNN_MKL': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=OFF '
        '-DUSE_CUDNN=OFF '
        '-DUSE_OPENCV=ON '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=mkl '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DUSE_ONEDNN=ON '
        '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_MKL': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=OFF '
        '-DUSE_CUDNN=OFF '
        '-DUSE_OPENCV=ON '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=mkl '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DUSE_ONEDNN=OFF '
        '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_GPU': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=ON '
        '-DUSE_CUDNN=ON '
        '-DUSE_OPENCV=ON  '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=open '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DMXNET_CUDA_ARCH="5.2 7.5" '
        '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_GPU_ONEDNN': (
        '-DCMAKE_C_COMPILER=cl '
        '-DCMAKE_CXX_COMPILER=cl '
        '-DUSE_CUDA=ON '
        '-DUSE_CUDNN=ON '
        '-DUSE_OPENCV=ON '
        '-DUSE_OPENMP=ON '
        '-DUSE_BLAS=open '
        '-DUSE_LAPACK=ON '
        '-DUSE_DIST_KVSTORE=OFF '
        '-DMXNET_CUDA_ARCH="5.2 7.5" '
        '-DUSE_ONEDNN=ON '
        '-DCMAKE_BUILD_TYPE=Release')

}


def windows_build(args):
    logging.info(f"Using vcvars environment:\n{args.vcvars}")
    if args.vcvars_ver:
        logging.info(f"Using vcvars version:\n{args.vcvars_ver}")

    path = args.output

    mxnet_root = get_mxnet_root()
    logging.info(f"Found MXNet root: {mxnet_root}")

    # cuda thrust / CUB + VS 2019 is flaky: try multiple times if fail
    MAXIMUM_TRY = 1
    build_try = 0

    while build_try < MAXIMUM_TRY:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        with remember_cwd():
            os.chdir(path)
            env = os.environ.copy()
            if 'GPU' in args.flavour:
                env["CXXFLAGS"] = '/FS /MD /O2 /Ob2'
            if not args.vcvars_ver:
                cmd = f'\"{args.vcvars}\" && cmake -GNinja {CMAKE_FLAGS[args.flavour]} {mxnet_root}'
            else:
                cmd = f'\"{args.vcvars}\" -vcvars_ver={args.vcvars_ver} && cmake -GNinja {CMAKE_FLAGS[args.flavour]} {mxnet_root}'
            logging.info(f"Generating project with CMake:\n{cmd}")
            check_call(cmd, shell=True, env=env)

            if not args.vcvars_ver:
                cmd = f'\"{args.vcvars}\" && ninja'
            else:
                cmd = f'\"{args.vcvars}\" -vcvars_ver={args.vcvars_ver} && ninja'
            logging.info(f"Building:\n{cmd}")

            t0 = int(time.time())
            ret = call(cmd, shell=True)


            if ret != 0:
                build_try += 1
                logging.info(f"{build_try} build(s) have failed")
            else:
                logging.info(
                    f'Build flavour: {args.flavour} complete in directory: \"{os.path.abspath(path)}\"'
                )
                logging.info("Build took {}".format(datetime.timedelta(seconds=int(time.time() - t0))))
                break

    if ret == 0:
        windows_package(args)
    else:
        logging.info("Build failed")
        sys.exit(1)


def windows_package(args):
    pkgfile = 'windows_package.7z'
    pkgdir = os.path.abspath('windows_package')
    logging.info("Packaging libraries and headers in package: %s", pkgfile)
    j = os.path.join
    pkgdir_lib = os.path.abspath(j(pkgdir, 'lib'))
    with remember_cwd():
        os.chdir(args.output)
        logging.info("Looking for static libraries and dlls in: \"%s", os.getcwd())
        libs = list(glob.iglob('**/*.lib', recursive=True))
        dlls = list(glob.iglob('**/*.dll', recursive=True))
        os.makedirs(pkgdir_lib, exist_ok=True)
        for lib in libs:
            logging.info("packing lib: %s", lib)
            shutil.copy(lib, pkgdir_lib)
        for dll in dlls:
            logging.info("packing dll: %s", dll)
            shutil.copy(dll, pkgdir_lib)
        os.chdir(get_mxnet_root())
        logging.info('packing python bindings')
        copy_tree('python', j(pkgdir, 'python'))
        logging.info('packing headers')
        copy_tree('include', j(pkgdir, 'include'))
        logging.info("Compressing package: %s", pkgfile)
        check_call(['7z', 'a', pkgfile, pkgdir])


def nix_build(args):
    path = args.output
    os.makedirs(path, exist_ok=True)
    with remember_cwd():
        os.chdir(path)
        logging.info("Generating project with CMake")
        check_call("cmake \
            -DUSE_CUDA=OFF \
            -DUSE_BLAS=open \
            -DUSE_OPENCV=OFF \
            -DUSE_OPENMP=OFF \
            -DCMAKE_BUILD_TYPE=Debug \
            -GNinja ..", shell=True)
        check_call("ninja", shell=True)


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logging.info("MXNet Windows build helper")
    if instance_info := ec2_instance_info():
        logging.info("EC2: %s", instance_info)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",
        help="output directory",
        default='build',
        type=str)

    parser.add_argument("--vcvars",
        help="vcvars batch file location, typically inside vs studio install dir",
        default=KNOWN_VCVARS['VS 2019'],
        type=str)

    parser.add_argument("--vcvars_ver",
        help="Optionally specifies the Visual Studio compiler toolset to use.\
            By default, the environment is set to use the current Visual Studio compiler toolset.",
        default=None,
        type=str)

    parser.add_argument("--arch",
        help="architecture",
        default='x64',
        type=str)

    parser.add_argument("-f", "--flavour",
        help="build flavour",
        default='WIN_CPU',
        choices=[x.name for x in BuildFlavour],
        type=str)

    args = parser.parse_args()
    logging.info("Build flavour: %s", args.flavour)

    system = platform.system()
    if system == 'Windows':
        logging.info("Detected Windows platform")
        if 'OpenBLAS_HOME' not in os.environ:
            os.environ["OpenBLAS_HOME"] = "C:\\Program Files\\OpenBLAS-v0.2.19"
        if 'OpenCV_DIR' not in os.environ:
            os.environ["OpenCV_DIR"] = "C:\\Program Files\\OpenCV-v3.4.1\\build"
        if 'CUDA_PATH' not in os.environ:
            os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2"
        if 'MKLROOT' not in os.environ:
            os.environ["MKLROOT"] = "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\mkl"
        windows_build(args)

    elif system in ['Linux', 'Darwin']:
        nix_build(args)

    else:
        logging.error(f"Don't know how to build for {platform.system()} yet")

    return 0


if __name__ == '__main__':
    sys.exit(main())
