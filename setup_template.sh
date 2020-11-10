#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

VERSION=${1:-"3.8"}

poetry build
rm -rf test_template
mkdir test_template
cd test_template
poetry init --name test --description test --author none --python ${VERSION} -n
poetry add ../dist/workflow-0.0.0-py3-none-any.whl
echo "test
test
0.1.0
test project
y" | poetry run python -m workflow.setup_project
poetry install
