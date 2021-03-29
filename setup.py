from setuptools import setup, find_packages

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['pyomo',
                         'coverage',
                         'numpy',
                         'scipy',
                         'pandas',
                         'assimulo'],
    'scripts': [],
    'include_package_data': True
}

setup(name='PharmaPy',
      version='0.0.0',
      packages=find_packages(),
      author='',
      **setuptools_kwargs)
