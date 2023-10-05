from setuptools import setup, find_packages, Extension

# Read the content of requirements.txt into a list
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(name='PharmaPy',
      version='0.0.1',
      packages=find_packages(),
      author='Daniel Casas-Orozco',
      author_email='dcasasor@purdue.edu',
      license='',
      url='',
      py_modules=["PharmaPy"],
      install_requires=requirements)