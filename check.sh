#!/bin/bash
set -eux

black ./*.py pytoolkit.git/pytoolkit

flake8 ./*.py pytoolkit.git/pytoolkit

mypy ./*.py pytoolkit.git/pytoolkit

pushd pytoolkit.git/docs/
./update.sh
make html
popd

#pyright ./*.py pytoolkit.git/pytoolkit

pylint --rcfile=pytoolkit.git/.pylintrc -j4 ./*.py pytoolkit.git/pytoolkit

pushd pytoolkit.git
pytest
popd
