#!python
#cython: language_level=3, boundscheck=False
from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules = cythonize("ORF_cython.pyx")
#    include_dirs=[numpy.get_include()]
)
