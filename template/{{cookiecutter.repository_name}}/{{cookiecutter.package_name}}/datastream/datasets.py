import numpy as np
import pandas as pd
from datastream import Datastream

from {{cookiecutter.package_name}} import problem


def datasets():
    datasets = problem.datasets()
    datasets['train'] = datasets['train'].split(
        key_column='index',
        proportions=dict(gradient=0.8, early_stopping=0.2),
        stratify_column='class_name',
        filepath='{{cookiecutter.package_name}}/splits/early_stopping.json',
    )

    return dict(
        gradient=datasets['train']['gradient'],
        early_stopping=datasets['train']['early_stopping'],
        compare=datasets['compare'],
    )
