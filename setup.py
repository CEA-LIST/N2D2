#!/usr/bin/env python3
""" Neural Network Design & Deployment with Python

N2D2 is a Python package to use the N2D2 engine with a Python interface.
"""

DOCLINES = (__doc__ or '').split("\n")

import sys
import os

# Python supported version checks
if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")



CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Typing :: Typed
Operating System :: POSIX
"""

import shutil
import pathlib
import subprocess
import multiprocessing

from math import ceil

from setuptools import setup, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext


def get_n2d2_version() -> str:
    n2d2_root = pathlib.Path().absolute()
    version = open(n2d2_root / "version.txt", "r").read().strip()
    return version

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build N2D2")

        # This lists the number of processors available on the machine
        # The compilation will use half of them
        max_jobs = str(ceil(multiprocessing.cpu_count() / 2))

        cwd = pathlib.Path().absolute()

        build_temp = cwd / "build"
        if not build_temp.exists():
            build_temp.mkdir(parents=True, exist_ok=True)

        build_lib = pathlib.Path(self.build_lib)
        if not build_lib.exists():
            build_lib.mkdir(parents=True, exist_ok=True)

        os.chdir(str(build_temp))

        self.spawn(['cmake', str(cwd)])
        if not self.dry_run:
            self.spawn(['make', '-j', max_jobs])
  
        os.chdir(str(cwd))

        ext_lib = build_temp / "lib"

        # Copy all shared object files from build_temp/lib to build_lib
        for root, dirs, files in os.walk(ext_lib.absolute()):
            for file in files:
                if file.endswith('.so'):
                    currentFile=os.path.join(root, file)
                    shutil.copy(currentFile, str(build_lib.absolute()))   

        # Copy export folder in "n2d2"
        shutil.copytree(
            str(cwd / "export"), 
            str(build_lib.absolute() / "n2d2" / "export")
        )        


if __name__ == '__main__':
    n2d2_packages = find_packages(where="./python")

    setup(
        name='n2d2',
        version=get_n2d2_version(),
        url='https://github.com/CEA-LIST/N2D2',
        license='CECILL-2.1',
        author='N2D2 Team',
        author_email='n2d2-contact@cea.fr',
        python_requires='>=3.7',
        description=DOCLINES[0],
        long_description_content_type="text/markdown",
        long_description="\n".join(DOCLINES[2:]),
        keywords=['n2d2', 'machine', 'learning'],
        project_urls={
            "Bug Tracker": "https://github.com/CEA-LIST/N2D2/issues",
            "Documentation": "https://cea-list.github.io/N2D2-docs/",
            "Source Code": "https://github.com/CEA-LIST/N2D2",
        },
        classifiers=[c for c in CLASSIFIERS.split('\n') if c],
        platforms=["Linux"],
        packages=n2d2_packages,
        package_dir={
            "n2d2": "python/n2d2",
            "pytorch_to_n2d2": "python/pytorch_to_n2d2",
            "keras_to_n2d2": "python/keras_to_n2d2",
        },
        ext_modules=[CMakeExtension('N2D2')],
        cmdclass={
            'build_ext': CMakeBuild,
        },

    )
