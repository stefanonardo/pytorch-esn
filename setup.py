from setuptools import setup, find_packages

setup(name='pytorch-esn',
      version='1.2.3',
      packages=find_packages(),
      install_requires=[
          'torch',
          'torchvision',
          'numpy'
      ],
      description="Echo State Network module for PyTorch.",
      author='Stefano Nardo',
      author_email='stefano_nardo@msn.com',
      license='MIT',
      url="https://github.com/stefanonardo/pytorch-esn"
      )
