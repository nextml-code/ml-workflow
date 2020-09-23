===========
ML Workflow
===========

.. image:: https://badge.fury.io/py/ml-workflow.svg
       :target: https://badge.fury.io/py/ml-workflow

ML workflow contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools and missing features to help bridge the gap
between frameworks and libraries that we utilize.

The main packages and tools that we build around are:

- pytorch
- ignite
- pytorch-datastream
- guild

See the `documentation <https://ml-workflow.readthedocs.io/en/latest/>`_
for more information.

Install in existing project
===========================

.. code-block::

    pip install ml-workflow

Create new project with MNIST template
======================================

.. code-block::

    mkdir new-project
    cd new-project
    virtualenv venv -p python3.8
    source venv/bin/activate
    pip install ml-workflow
    python -m workflow.setup_project

    pip install -r requirements.txt
    pip install -r dev_requirements.txt
    pip freeze > dev_requirements.txt

    # reactivate environment to find guild
    deactivate
    source venv/bin/activate

You can train a model and inspect the training with:

.. code-block::

    guild run prepare
    guild run train
    guild tensorboard


Development
===========

Prepare and run tests

.. code-block::

    git clone git@github.com:aiwizo/ml-workflow.git
    cd ml-workflow
    virtualenv venv --python python3.8
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r dev_requirements.txt
    pip install pytest
    python -m pytest

Test template
=============

.. code-block::

    ./setup_template.py
    ./test_template.py

Use development version in project
==================================

The following steps will create a link to the local directory and any changes 
made to the package there will directly carry over to your project environment.

.. code-block::

    cd path/to/my/project
    source venv/bin/activate

    cd path/to/work/area
    git clone git@github.com:aiwizo/ml-workflow.git
    cd ml-workflow
    pip install -e .
