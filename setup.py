"""Setup script."""

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

setup(
    name='evtrans',
    version="0.1.0",

    author="Yuhuang Hu",
    author_email="yuhuang.hu@ini.uzh.ch",

    packages=["evtrans"],

    classifiers=list(filter(None, classifiers.split('\n'))),
)
