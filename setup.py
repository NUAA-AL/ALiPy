#!/usr/bin/env python

from setuptools import setup

setup(
    name='alipy',
    version='1.0.0',
    description='Active Learning in Python',
    long_description=open('README.md', encoding='UTF-8').read(),
    author='Ying-Peng Tang, Guo-Xiang Li, Sheng-Jun Huang',
    author_email='tangyp@nuaa.edu.cn, GuoXiangLi@nuaa.edu.cn, huangsj@nuaa.edu.cn',
    url='https://github.com/NUAA-AL/ALiPy',
    setup_requires=[],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'prettytable'],
    packages=[
        'alipy',
        'alipy.data_manipulate',
        'alipy.experiment',
        'alipy.index',
        'alipy.metrics',
        'alipy.oracle',
        'alipy.query_strategy',
        'alipy.utils',
    ],
    package_dir={
        'alipy': 'alipy',
        'alipy.data_manipulate': 'alipy/data_manipulate',
        'alipy.experiment': 'alipy/experiment',
        'alipy.index': 'alipy/index',
        'alipy.metrics': 'alipy/metrics',
        'alipy.oracle': 'alipy/oracle',
        'alipy.query_strategy': 'alipy/query_strategy',
        'alipy.utils': 'alipy/utils',
    },
)
