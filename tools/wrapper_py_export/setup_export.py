from setuptools import Extension, setup
import os
import sys
import glob
import shutil

from Cython.Build import cythonize

# Get save_outputs option from user
opt_flag = []
if "--save_outputs" in sys.argv:
    opt_flag = ["-DSAVE_OUTPUTS"]
    sys.argv.remove("--save_outputs")

extensions = [
    Extension("n2d2_export", ["n2d2_export.pyx"],
        include_dirs=["./include", "./dnn/include"],
        extra_compile_args=["-std=c++11", "-fopenmp", "-O3", "-fsigned-char"] + opt_flag,
        extra_link_args=["-fopenmp"]),
]


if __name__ == "__main__":

    # Build library
    setup(
        ext_modules=cythonize(extensions, language_level=3)
    )

    # Remove residual files from build process
    os.remove("n2d2_export.cpp")
    shutil.rmtree("build")

    # Rename lib
    for file in glob.glob("n2d2_export.*.so"):
        shutil.move(file, "n2d2_export.so")

