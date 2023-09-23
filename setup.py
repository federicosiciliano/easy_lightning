import os

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

setup(
    name='exp_utils',
    packages=['exp_utils'],
    description='Experiments utils',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.1',
    install_requires=install_requires,
    url='https://github.com/siciliano-diag/exp_utils.git',
    author='siciliano-diag',
    author_email='siciliano@diag.uniroma1.it',
    keywords=['pip','MachineLearning']
    )




#license='MIT',
#project_urls = {"Bug Tracker": "https://github.com/mike-huls/toolbox/issues"},

