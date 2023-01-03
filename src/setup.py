from __future__ import print_function
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = cythonize([
    Extension("cython_transforms", 
             ["cython/cython_transforms.pyx", "cython/MurmurHash3.cpp"],
              include_dirs=[numpy.get_include()]
              )
], annotate=True)

setup(
    name='Hello world app',
    ext_modules=extensions,
    zip_safe=False,
)