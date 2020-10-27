#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

VERSION=${1:-"3.8"}

rm -rf test_template
mkdir test_template
cd test_template
poetry init
poetry run pip install -e ../.
echo "test
test
0.1.0
test project
y" | poetry run python -m workflow.setup_project
poetry install
