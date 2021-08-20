from os.path import join, dirname
from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
import numpy as np


path = dirname(__file__)
src_dir = join(dirname(path), "..", "src")
defs = [("NPY_NO_DEPRECATED_API", 0)]
inc_path = np.get_include()
# If we need random/lib library from numpy...
lib_path = join(np.get_include(), "..", "..", "random", "lib")

extension = Extension(
    "extensions",
    sources=[join(".", "extensions.pyx")],
    include_dirs=[
        np.get_include(),
        join(path, "..", "..")
    ],
    define_macros=defs
)

setup(
    ext_modules=cythonize(extension,
                          language_level="3"),
    zip_safe=False
)
