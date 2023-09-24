import os
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

setup(
    name='your_combined_project_name',  # Replace with your desired project name
    packages=['exp_utils', 'data_utils'],  # List all packages from both projects
    description='Combined Description',  # Replace with a combined project description
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.1',
    install_requires=install_requires,
    url='https://github.com/your-github-url',  # Replace with your GitHub repository URL
    author='siciliano-diag',
    author_email='siciliano@diag.uniroma1.it',
    keywords=['pip', 'MachineLearning']
)
