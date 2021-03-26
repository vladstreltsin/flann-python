from distutils.core import Extension, setup
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from distutils.command.build import build as DistutilsBuild
from distutils.command.install import install
from distutils.sysconfig import customize_compiler
import subprocess
import os.path as osp
import os
import numpy as np
from third_party.FLANN import FLANN_DIR, FLANN_BUILD_DIR, FLANN_SRC_DIR
import shutil

ROOT_DIR = osp.dirname(__file__)


# Dispose of the compiler warning
# cc1plus: warning: command line option “-Wstrict-prototypes” is valid for Ada/C/ObjC but not for C++
# Taken from:
# https://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
class BuildExt(build_ext):

    def __init__(self, *args, **kwargs):
        super(BuildExt, self).__init__(*args, **kwargs)

    def _build_flann(self):
        shutil.rmtree(FLANN_BUILD_DIR, ignore_errors=True)
        current_dir = os.getcwd()
        os.makedirs(FLANN_BUILD_DIR, exist_ok=True)
        os.chdir(f'{FLANN_BUILD_DIR}')
        subprocess.call(['cmake',
                         # f'-DCMAKE_INSTALL_PREFIX=${FLANN_LIB_DIR}',
                         '-DBUILD_EXAMPLES=OFF',
                         f'-DBUILD_PYTHON_BINDINGS=OFF',
                         '-DBUILD_DOC=OFF',
                         '-DBUILD_TESTS=OFF',
                         f'{FLANN_DIR}'])
        subprocess.call(['make'])
        os.chdir(current_dir)

    def build_extensions(self):
        self._build_flann()
        customize_compiler(self.compiler)

        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")

        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


exts = [

        # The FLANN extension
        Extension(name='flann.flann',
                  sources=[os.path.join(ROOT_DIR, 'cy', 'flann.pyx')],
                  library_dirs=[os.path.join(FLANN_BUILD_DIR, 'lib')],
                  runtime_library_dirs=[os.path.join(FLANN_BUILD_DIR, 'lib')],
                  libraries=['flann'],
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")],
                  include_dirs=[os.path.join(FLANN_SRC_DIR, 'cpp'),
                                np.get_include()],
                  language="c"),
        ]


setup(name='flann-python',
      cmdclass={'build_ext': BuildExt},
      ext_modules=cythonize(exts,
                            compiler_directives={'language_level': 3},
                            build_dir='build',
                            ),
      packages=['flann'],
      zip_safe=False)
