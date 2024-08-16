from setuptools import find_packages, setup

find_packages()
setup(
    name="src",
    packages=find_packages(),
    version="1.0.0",
    description="AUTOENCODIX is a framework for multi-modal data integration by autoencoders.",
    author="Max Joas, Jan Ewald",
    license="MIT",
)
