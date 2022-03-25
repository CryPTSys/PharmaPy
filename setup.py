from setuptools import setup, find_packages

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['coverage',
                         'numpy',
                         'scipy',
                         'pandas',
                         'assimulo',
                         'cyipopt'],
    'scripts': [],
    'include_package_data': True
}

setup(name='PharmaPy',
      version='0.0.0',
      packages=find_packages(),
      author='Daniel Casas-Orozco',
      author_email='dcasasor@purdue.edu',
      license='',
      url='',
      **setuptools_kwargs)
