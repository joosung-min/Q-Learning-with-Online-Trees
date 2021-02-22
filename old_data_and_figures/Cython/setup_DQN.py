from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("DQN.pyx", language=3)
)