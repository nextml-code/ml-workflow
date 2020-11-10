#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

cd test_template
shopt -s extglob 
rm -rf .github
rm -rf !(pyproject.toml)
echo "repository
package
0.1.0
project description
y" | poetry run python -m workflow.setup_project
poetry run guild run prepare -y
poetry run guild run search_lr n_batches=10 -y
poetry run guild run train max_epochs=2 n_batches_per_epoch=2 -y
poetry run guild run retrain max_epochs=1 n_batches_per_epoch=1 -y
poetry run guild run evaluate -y
