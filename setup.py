import os

from setuptools import find_packages, setup


HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the requirements
with open("requirements.txt", "r", encoding="utf-8") as requirement_file:
    requirements = requirement_file.readlines()

setup(
    name="vietac",
    version="0.1.0",
    description="Vietnamese Auto Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="hungtx",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.7",
    py_modules=["vietac"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Artificial Intelligence",
    ],
)
