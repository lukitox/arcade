#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Lukas Hilbers",
    author_email='l.hi@posteo.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Describe aircraft configurations parametrically with pydantic.",
    entry_points={
        'console_scripts': [
            'arcade=arcade.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='arcade',
    name='arcade',
    packages=find_packages(include=['arcade', 'arcade.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lukitox/arcade',
    version='0.1.0',
    zip_safe=False,
)
