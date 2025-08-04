from setuptools import setup, find_packages

setup(
    name="nnforhjb",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "loguru>=0.6.0",
    ],
) 