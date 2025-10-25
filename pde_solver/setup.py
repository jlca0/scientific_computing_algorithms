from setuptools import setup, find_packages

setup(
    name='pde_solver',
    version='0.1',
    packages=find_packages(include=['solver', 'solver.*']),
)
