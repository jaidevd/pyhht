#!/usr/bin/env bash
# This script is inspired from **pgmpy** implementation of continous test
# integration. This is meant to "install" all the packages required for installing
# semantic.

# License: The MIT License (MIT)

set -e

apt-get update -qq
apt-get install build-essential -qq

if [[ "$DISTRIB" == "conda" ]]; then
	# Deactivate the travis-provided virtual environment and setup a
	# conda-based environment instead
	deactivate

	# Use the miniconda installer for faster download / install of conda
	# itself
	wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
		-O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    hash -r
	conda config --set always_yes yes --set changeps1 no
	conda update conda
	conda info -a

	conda create -n testenv python=$PYTHON_VERSION --file ci/requirements.txt
    source activate testenv
    pip install git+git://github.com/scikit-signal/pytftb@master
fi

if [[ "$COVERAGE" == "true" ]]; then
	pip install coverage coveralls
fi

# Build pgmpy
python setup.py develop
