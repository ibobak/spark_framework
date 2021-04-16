from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line for line in f if not line.startswith("#")]

setup(
    name="spark_framework",
    version="1.0",
    description="collection of useful functions to make your PySpark code more convenient and robust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ihor Bobak",
    author_email="ibobak@gmail.com",
    license="Apache 2.0",
    packages=find_packages(exclude=["tests.*", "tests"]),
    platforms=["any"],
    install_requires=requirements,
)
