import os
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

setup(
    name='Easy Lightning', 
    packages=['exp_utils', 'data_utils', 'torch_utils'],  # List all packages from all projects
    description='Easy Lightning: Simplify AI-Deep learning with PyTorch Lightning',  
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.1',
    install_requires=install_requires,
    url='https://github.com/fed21',  # Replace with your GitHub repository URL
    author='Federico Siciliano, Federico Carmignani',
    author_email='siciliano@diag.uniroma1.it, carmignanifederico@gmail.com',
    keywords=['DeepLearning', 'MachineLearning', 'PyTorch', 'Lightning', 'AI']
)
