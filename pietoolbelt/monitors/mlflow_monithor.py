import os
import subprocess
from random import random
from time import time
from typing import List

import numpy as np

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags

from piepline.train_config.metrics import MetricsGroup, AbstractMetric
from piepline.monitoring.monitors import AbstractMetricsMonitor, AbstractLossMonitor

__all__ = ['MLFlowMonitor']


def _already_ran(git_commit, client, experiment_id=None):
    """
    This code was get and modifyed from mlflow multiprocess example: https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            continue
        return client.get_run(run_info.run_id)
    return None


class MLFlowMonitor(AbstractMetricsMonitor, AbstractLossMonitor):
    def __init__(self, server_url: str, project_name: str, run_name: str = None):
        AbstractMetricsMonitor.__init__(self)
        AbstractLossMonitor.__init__(self)

        self._prefix = ''

        with mlflow.start_run() as active_run:
            git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        self._client = MlflowClient(tracking_uri=server_url)
        experiment = self._client.get_experiment_by_name(name=project_name)

        self._run = None
        if experiment is not None:
            self._run = _already_ran(git_commit, client=self._client, experiment_id=experiment.experiment_id)

        if self._run is None:
            self._run = self._client.create_run(experiment.experiment_id, tags={mlflow_tags.MLFLOW_GIT_COMMIT: git_commit})

        self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_SOURCE_NAME, os.path.basename(__file__))

        sproc = subprocess.Popen("git rev-parse --symbolic-full-name HEAD", shell=True, stdout=subprocess.PIPE)
        branch_match = sproc.stdout.read()

        if branch_match == "HEAD":
            branch = None
        else:
            branch = os.path.basename(branch_match.decode().strip())

        if branch is not None:
            self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_GIT_BRANCH, branch)
            if run_name is None:
                self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_RUN_NAME, branch)
        else:
            if run_name is not None:
                self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_RUN_NAME, branch)
        self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_USER, os.getlogin())

    def _log_metric(self, name: str, value: float, epoch_idx: int = None):
        self._client.log_metric(self._run.info.run_id, key=self._prefix + name, value=value, timestamp=int(time() * 1000),
                                step=self._epoch_num if epoch_idx is None else epoch_idx)

    def set_prefix(self, prefix: str) -> 'MLFlowMonitor':
        self._prefix = prefix + '/'
        return self

    def update_losses(self, losses: {}) -> None:
        """
        Update monitor

        :param losses: losses values with keys 'train' and 'validation'
        """

        def on_loss(name: str, values: np.ndarray or dict) -> None:
            if isinstance(values, dict):
                for k, v in values.items():
                    self._log_metric('loss_{}/{}'.format(name, k), value=np.mean(v))
            else:
                self._log_metric('loss/{}'.format(name), np.mean(values))

        self._iterate_by_losses(losses, on_loss)

    def _process_metric(self, path: List[MetricsGroup], metric: AbstractMetric):
        """
        Update console

        :param metrics: metrics
        """

        tag = '/'.join([p.name() for p in path] + [metric.name()])

        values = metric.get_values().astype(np.float32)
        if values.size > 0:
            self._log_metric(tag, float(np.mean(values)))

    def update_scalar(self, name: str, value: float, epoch_idx: int = None) -> None:
        """
        Update scalar on tensorboard

        :param name: the classic tag for TensorboardX
        :param value: scalar value
        :param epoch_idx: epoch idx. If doesn't set - use last epoch idx stored in this class
        """
        self._log_metric(name, value, epoch_idx=epoch_idx)

    def set_user_name(self, user_name: str) -> None:
        self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_USER, user_name)

    def add_note(self, note_text: str) -> None:
        self._client.set_tag(self._run.info.run_id, mlflow_tags.MLFLOW_RUN_NOTE, note_text)

    def log_params(self, params_dict: dict) -> None:
        for key, value in params_dict.items():
            self._client.log_param(self._run.info.run_id, key, value)

    def close(self):
        self._client.set_terminated(self._run.info.run_id, status='FINISHED', end_time=int(time() * 1000))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
