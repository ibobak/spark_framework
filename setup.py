"""Setup of Spark Framework"""

from setuptools import setup

with open("requirements.txt") as f:
    requirements = [line for line in f if not line.startswith("#")]

setup(
    name="spark_framework",
    version="1.29",
    description="Alternative pythonic style API to work with Apache Spark",
    long_description="The project is a collection of useful functions "
                     "that allow to write PySpark code in a more convenient way",
    author="Ihor Bobak",
    author_email="ibobak@gmail.com",
    license="Apache 2.0",
    packages=["spark_framework"],
    platforms=["any"],
    install_requires=requirements,
)
