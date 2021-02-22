from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("ORF_cython.pyx", language=3)
)