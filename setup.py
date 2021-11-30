from setuptools import setup, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
import os
import pathlib

class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])

class build_ext(build_ext_orig):
    """Adapted from : https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py
    """
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)
        cmake_args = [
        ]
        # example of build args
        build_args = [
            '--', '-j4'
        ]
        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

if __name__ == '__main__':
    print("Looking for packages ...")
    n2d2_packages = find_packages(where="./python")
    packages = n2d2_packages

    setup(
        name='n2d2',
        version='0.0.1',
        package_dir={
            "n2d2": "python/n2d2",
			"pytorch_interoperability": "python/pytorch_interoperability"
        },
        packages=packages,
        ext_modules=[CMakeExtension('N2D2')],
        cmdclass={
            'build_ext': build_ext,
        },

    )
