import os
from setuptools import setup, Extension

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SPM_ROOT = os.path.join(PROJECT_ROOT, "build_spm", "sentencepiece")

ext_modules = [
    Extension(
        'sentencepiece_lite',
        ['sbp_module.cc', 'sentencepiece_lite.cc'],
        include_dirs=[
            '.',
            os.path.join(SPM_ROOT, 'build', 'third_party', 'flatbuffers-src', 'include'),
            os.path.join(SPM_ROOT, 'build', 'lite'),
            os.path.join(SPM_ROOT, 'third_party', 'abseil-cpp'),
        ],
        define_macros=[("Py_LIMITED_API", "0x030B0000")],
        extra_compile_args=['-O3', '-std=c++20', '-fPIC', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        language='c++',
        py_limited_api=True,
    ),
]

setup(
    name='sentencepiece_lite',
    version='0.1.0',
    description='SentencePiece Lite with Safe Boundary Pre-tokenization',
    ext_modules=ext_modules,
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
