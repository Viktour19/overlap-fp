from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    
    name="biasscan",
    ext_modules=cythonize(
        [
           Extension("biasscan.*", ["*.py"]),
           Extension("optim.*", ["optim/*.py"]),
           Extension("solver.*", ["solver/*.py"]),
           Extension("util.*", ["util/*.py"])
        ],
        build_dir="build",
        compiler_directives=dict(
        always_allow_keywords=True
        )),
    cmdclass=dict(
        build_ext=build_ext
    ),
    packages=["optim", "solver", "util"]
)