"""
Setup configuration for PyTorch-ESN package.

Echo State Network implementation for PyTorch with support for sequence modeling,
time series prediction, and classification tasks.
"""
import os
from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    """Read README.md for the long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read version from package
def get_version():
    """Get version from package __init__.py."""
    version_file = os.path.join(os.path.dirname(__file__), 'torchesn', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '1.2.5'  # Fallback version

setup(
    name='pytorch-esn',
    version=get_version(),
    packages=find_packages(),
    
    # Dependencies
    install_requires=[
        'torch>=2.8.0',
        'numpy>=1.16.0',
    ],
    
    # Optional dependencies for examples and development
    extras_require={
        'examples': [
            'torchvision>=0.2.0',
        ],
    },
    
    # Package metadata
    description="Echo State Network implementation for PyTorch with support for sequence modeling and time series prediction.",
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Author information
    author='Stefano Nardo',
    author_email='stefano_nardo@msn.com',
    
    # Project URLs
    url="https://github.com/stefanonardo/pytorch-esn",
    project_urls={
        'Bug Reports': 'https://github.com/stefanonardo/pytorch-esn/issues',
        'Source': 'https://github.com/stefanonardo/pytorch-esn',
        'Documentation': 'https://github.com/stefanonardo/pytorch-esn/blob/master/README.md',
    },
    
    # License and classifiers
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Package configuration
    python_requires='>=3.11',
    include_package_data=True,
    zip_safe=False,
    
    # Keywords for discovery
    keywords=[
        'pytorch', 'echo-state-network', 'esn', 'reservoir-computing',
        'time-series', 'sequence-modeling', 'recurrent-neural-network',
        'machine-learning', 'deep-learning'
    ],
)
