#!python
#cython: language_level=3, boundscheck=False
from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("ORF_py_cython.pyx")
#    include_dirs=[numpy.get_include()]
)
