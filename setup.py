from setuptools import setup, find_packages

setup(
    name='robo_copilot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        # Add other dependencies as needed
    ],
)
