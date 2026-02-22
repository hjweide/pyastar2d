import pathlib

from setuptools import Extension, find_packages, setup

# Use pathlib for paths
here = pathlib.Path(__file__).parent.resolve()

# Read README and requirements
long_description = (here / 'README.md').read_text(encoding='utf-8')
install_requires = (here / 'requirements.txt').read_text().splitlines()


class get_numpy_include:
    """Defer numpy import until it is actually installed."""

    def __str__(self):
        import numpy

        return numpy.get_include()


# Define the C++ extension
astar_module = Extension(
    name='pyastar2d.astar',
    sources=[
        'src/cpp/astar.cpp',
        'src/cpp/experimental_heuristics.cpp',
    ],
    define_macros=[
        ('Py_LIMITED_API', '0x03090000'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_21_API_VERSION'),
        ('NPY_TARGET_VERSION', 'NPY_1_21_API_VERSION'),
    ],
    py_limited_api=True,
    include_dirs=[
        'src/cpp',
        get_numpy_include(),
    ],
    extra_compile_args=['-O3', '-fpic', '-Wall'],
    language='c++',
)

# Define package metadata
setup(
    name='pyastar2d',
    use_scm_version=True,
    author='Hendrik Weideman',
    author_email='hjweide@gmail.com',
    description='A simple implementation of the A* algorithm for path-finding on a two-dimensional grid.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hjweide/pyastar2d',
    packages=find_packages(where='src', exclude=('tests',)),
    package_dir={'': 'src'},
    install_requires=install_requires,
    python_requires='>=3.9',
    ext_modules=[astar_module],
    options={'bdist_wheel': {'py_limited_api': 'cp39'}},
)
