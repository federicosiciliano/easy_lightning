import os
from setuptools import setup, find_packages

# Read the contents of the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Check if requirements.txt files exists and if so, read their contents
colab_required = []
if os.path.isfile("colab_requirements.txt"):
    with open('colab_requirements.txt') as f:
        colab_required = f.read().splitlines()

install_requires = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

# submodules = ['easy_data @ git+https://github.com/PokeResearchLab/easy_data.git',
#               'easy_exp @ git+https://github.com/PokeResearchLab/easy_exp.git',
#               'easy_rec @ git+https://github.com/PokeResearchLab/easy_rec.git',
#               'easy_torch @ git+https://github.com/PokeResearchLab/easy_torch.git']

# Define the package setup configuration
setup(
    name='Easy Lightning',  # Replace with your package name
    packages = find_packages(),  # List of all packages included in your project
    description='Easy Lightning: Simplify AI-Deep learning with PyTorch Lightning',  
    long_description=long_description,  # Use the contents of README.md as the long description
    long_description_content_type="text/markdown",
    version='1.0.0',  # Specify the version of your package
    #install_requires=submodules,  # List of required dependencies
    extras_require = {'all': install_requires,
                      'colab': colab_required},
    url='https://github.com/PokeResearchLab/easy_lightning.git',  # Replace with the URL of your GitHub repository
    author='???',
    author_email='???',
    keywords=['DeepLearning', 'MachineLearning', 'PyTorch', 'Lightning', 'AI']  # Keywords related to your package
)
