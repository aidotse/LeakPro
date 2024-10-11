"""Setuptools configuration for the package."""

from setuptools import find_packages, setup


# Function to read dependencies from the file
def parse_requirements(filename:str)->list:
    """Parse the requirements from a file.

    Args:
    ----
        filename (str): The name of the file to parse.

    Returns:
    -------
        list: A list of parsed requirements.

    """
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="leakpro",             # Name of your package
    version="0.1",               # Package version
    packages=find_packages(),    # Automatically find package modules
    install_requires=parse_requirements("requirements.txt"),         # List of dependencies
    author="LeakPro Team",      # Author name
    author_email="johan.ostman@ai.se",
    license="Apache 2.0",              # License type
    description="A package for privacy risk analysis.",  # Short description
    url="https://github.com/aidotse/LeakPro",
)
