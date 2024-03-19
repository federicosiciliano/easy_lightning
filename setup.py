import os
from setuptools import setup, find_packages

# Read the contents of the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Check if a requirements.txt file exists and if so, read its contents
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

# Define the package setup configuration
setup(
    name='Easy Lightning',  # Replace with your package name
    packages = find_packages(),  # List of all packages included in your project
    description='Easy Lightning: Simplify AI-Deep learning with PyTorch Lightning',  
    long_description=long_description,  # Use the contents of README.md as the long description
    long_description_content_type="text/markdown",
    version='0.0.1',  # Specify the version of your package
    install_requires=install_requires+['data_utils @ git+https://github.com/siciliano-diag/data_utils.git',
                                       'exp_utils @ git+https://github.com/siciliano-diag/exp_utils.git',
                                       'rec_utils @ git+https://github.com/siciliano-diag/rec_utils.git',
                                       'torch_utils @ git+https://github.com/siciliano-diag/torch_utils.git'],  # List of required dependencies
    url='https://github.com/siciliano-diag/easy_lightning.git',  # Replace with the URL of your GitHub repository
    author='Federico Siciliano, Federico Carmignani',
    author_email='siciliano@diag.uniroma1.it, carmignanifederico@gmail.com',
    keywords=['DeepLearning', 'MachineLearning', 'PyTorch', 'Lightning', 'AI']  # Keywords related to your package
)