import setuptools
from distutils.core import Extension
from setuptools import dist
dist.Distribution().fetch_build_eggs(["numpy"])
import numpy

astar_module = Extension(
    'pyastar2d.astar', sources=['src/cpp/astar.cpp', 'src/cpp/experimental_heuristics.cpp'],
    include_dirs=[
                    numpy.get_include(),   # for numpy/arrayobject.h
                    'src/cpp'    # for experimental_heuristics.h
                 ],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)


with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyastar2d",
    version="1.0.6",
    author="Hendrik Weideman",
    author_email="hjweide@gmail.com",
    description=(
        "A simple implementation of the A* algorithm for "
        "path-finding on a two-dimensional grid."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hjweide/pyastar2d",
    install_requires=install_requires,
    packages=setuptools.find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    ext_modules=[astar_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
