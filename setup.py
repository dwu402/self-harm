from setuptools import find_packages, setup
import sys

sys.path[0:0] = ['src']

setup(
    name='self-harm',
    version="0.0.1",
    description="Generalised Profiling for Parameter Estimation of ODEs",
    author="David Wu",
    author_email="dwu402@aucklanduni.ac.nz",
    packages=['selfharm', 'selfharm.functions']
)
