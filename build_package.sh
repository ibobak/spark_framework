#!/bin/sh
rm -r dist
python setup.py bdist_wheel
rm -r build
rm -r spark_framework.egg-info
