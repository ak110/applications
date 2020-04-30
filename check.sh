#!/bin/bash
set -eux

black ./*.py pytoolkit

flake8 ./*.py pytoolkit

mypy ./*.py pytoolkit

pushd pytoolkit/docs/
./update.sh
make html
popd

#pyright ./*.py pytoolkit

pylint --rcfile=pytoolkit/.pylintrc -j4 ./*.py pytoolkit

pushd pytoolkit/
pytest
popd
