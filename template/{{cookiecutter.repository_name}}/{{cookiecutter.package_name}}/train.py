from functools import partial
from pathlib import Path
import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import ignite
import logging
import workflow
from workflow.functional import starcompose
from workflow.torch import set_seeds
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from datastream import Datastream

from {{cookiecutter.package_name}} import (
    datastream, architecture, metrics, log_examples
)


def train(config):
    set_seeds(config['seed'])
    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate']
    )

    train_state = dict(model=model, optimizer=optimizer)

    if Path('model').exists():
        print('Loading model checkpoint')
        workflow.ignite.handlers.ModelCheckpoint.load(
            train_state, 'model/checkpoints', device
        )
        workflow.torch.set_learning_rate(optimizer, config['learning_rate'])

    # 
    def process_batch(examples):
        predictions = model.predictions(
            architecture.FeatureBatch.from_examples(examples)
        )
        loss = predictions.loss(examples)
        return predictions, loss

    @workflow.torch.decorators.train(model, optimizer)
    def train_batch(examples):
        predictions, loss = process_batch(examples)
        loss.backward()
        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
        )

    @workflow.torch.decorators.evaluate(model)
    def evaluate_batch(examples):
        predictions, loss = process_batch(examples)
        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
        )

    evaluate_data_loaders = {
        f'evaluate_{name}': datastream.data_loader(
            batch_size=config['eval_batch_size'],
            num_workers=config['n_workers'],
            collate_fn=tuple,
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    gradient_data_loader = (
        datastream.GradientDatastream()
        .data_loader(
            batch_size=config['batch_size'],
            num_workers=config['n_workers'],
            n_batches_per_epoch=config['n_batches_per_epoch'],
            worker_init_fn=partial(worker_init, config['seed'], trainer),
            collate_fn=tuple,
        )
    )

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    gradient_metrics = metrics.gradient_metrics(tensorboard_logger)
    early_stopping = workflow.EarlyStopping(
        tensorboard_logger,
        lambda summaries: summaries['early_stopping']['accuracy'],
    )

    for epoch in range(config['max_epochs']):
        for examples in workflow.progress(
            gradient_data_loader, gradient_metrics[['loss', 'accuracy']]
        ):
            output = train_batch(examples)
            gradient_metrics.update_(output)
            gradient_metrics.log()
            # optional: schedule learning rate

        for name, data_loader in evaluate_data_loaders:

            metrics = metrics.evaluate_metrics(name, tensorboard_logger)
            for examples in tqdm(data_loader):
                output = evaluate_batch(examples)
                metrics.update_(output)

            metrics.log()

        improved, out_of_patience = early_stopping.score_(output)
        if improved:
            torch.save(train_state, 'model_checkpoint.pt')
        elif out_of_patience(output):
            break
