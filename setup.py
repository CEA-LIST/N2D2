#!/usr/bin/env python3
""" Neural Network Design & Deployment with Python.

N2D2 is a Python package to use the N2D2 engine with a Python interface.
"""

DOCLINES = (__doc__ or '').split("\n")
VERSION = '0.1.0'

import sys

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
Programming Language :: Python :: 3.10
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

import os
import shutil
import pathlib
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext


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
        max_jobs = str(multiprocessing.cpu_count() / 2)

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


if __name__ == '__main__':
    print("Looking for packages ...")
    n2d2_packages = find_packages(where="./python")
    packages = n2d2_packages

    setup(
        name='n2d2',
        version=VERSION,
        url='https://github.com/CEA-LIST/N2D2',
        description=DOCLINES[0],
        long_description_content_type="text/markdown",
        long_description="\n".join(DOCLINES[2:]),
        keywords=['python', 'AI', 'neural networks'],
        project_urls={
            "Bug Tracker": "https://github.com/CEA-LIST/N2D2/issues",
            "Documentation": "https://cea-list.github.io/N2D2-docs/",
            "Source Code": "https://github.com/CEA-LIST/N2D2",
        },
        license='CECILL-2.1',
        classifiers=[c for c in CLASSIFIERS.split('\n') if c],
        platforms=["Linux"],
        packages=packages,
        package_dir={
            "n2d2": "python/n2d2",
			"pytorch_interoperability": "python/pytorch_interoperability"
        },
        python_requires='>=3.7',
        ext_modules=[CMakeExtension('N2D2')],
        cmdclass={
            'build_ext': CMakeBuild,
        },

    )
