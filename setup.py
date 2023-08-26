import os
from setuptools import setup, find_packages

# from https://stackoverflow.com/a/9079062
import sys
if sys.version_info[0] < 3:
    raise Exception("micromanager only supports Python3. Did you run $python setup.py <option>.? "
                    "Try running $python3 setup.py <option>.")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='micro-manager-precice',
    version='v0.3.0rc2',
    description='micro-manager-precice is a package which facilitates two-scale macro-micro coupled simulations using preCICE',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://precice.org/tooling-micro-manager-overview.html',
    entry_points={
        'console_scripts': ['micro_manager=micro_manager.micro_manager:main']},
    author='Ishaan Desai',
    author_email='ishaan.desai@uni-stuttgart.de',
    license='LGPL-3.0',
    packages=find_packages(
        exclude=['examples']),
    install_requires=[
        'pyprecice==2.5.0.4',
        'numpy>=1.13.3',
        'mpi4py'],
    test_suite='tests',
    zip_safe=False)
