=============================
{{cookiecutter.package_name}}
=============================

Development
===========

Installation
------------

.. code-block::

    virtualenv venv --python python3.8
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r dev_requirements.txt
    guild run prepare

Training
--------

.. code-block::

    guild run prepare
    guild run train
    guild run retrain model=<model-hash>
    guild run evaluate model=<model-hash>
