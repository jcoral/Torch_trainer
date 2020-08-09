# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import


from setuptools import setup, find_packages

setup(
    name='torch-trainer',
    version='1.0.4',
    description='The torch trainer, has the same API with keras.',
    author='coral',
    license="GPLv3",
    keywords='torch trainer',
    url='https://github.com/jcoral/Torch_trainer',
    packages=find_packages(exclude=('test',)),
    install_requires=['pytorch-ignite>=0.2'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License (GPL)",
    ],
    zip_safe=False
)










