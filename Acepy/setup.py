#!/usr/bin/env python

from setuptools import setup

setup(
    name='acepy',
    version='0.1.1',
    description='Active learning tools in Python',
    long_description=open('README.md').read(),
    author='Ying-Peng Tang',
    author_email='',
    url='https://github.com/tangypnuaa/acepy',
    setup_requires=[],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'prettytable'],
    packages=[
        'acepy',
        'acepy.data_manipulate',
        'acepy.experiment',
        'acepy.index',
        'acepy.metrics',
        'acepy.oracle',
        'acepy.query_strategy',
        'acepy.utils',
    ],
    package_dir={
        'acepy': 'acepy',
        'acepy.data_manipulate': 'acepy/data_manipulate',
        'acepy.experiment': 'acepy/experiment',
        'acepy.index': 'acepy/index',
        'acepy.metrics': 'acepy/metrics',
        'acepy.oracle': 'acepy/oracle',
        'acepy.query_strategy': 'acepy/query_strategy',
        'acepy.utils': 'acepy/utils',
    },
)
