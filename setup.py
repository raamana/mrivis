#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'nibabel',
    'matplotlib',
    'numpy',
]

setup_requirements = [
    'pytest-runner',
    # TODO(raamana): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='mrivis',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Tools and scripts for visualization and comparison of 3d MRI scans (T1, T2 etc)",
    long_description=readme + '\n\n' + history,
    author="Pradeep Reddy Raamana",
    author_email='raamana@gmail.com',
    url='https://github.com/raamana/mrivis',
    packages=find_packages(include=['mrivis']),
    entry_points={
        'console_scripts': [
            'mrivis=mrivis.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='mrivis',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
