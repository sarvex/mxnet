# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import, unused-variable
"""Gluon Estimator Utility Functions"""

from ...loss import SoftmaxCrossEntropyLoss
from ...metric import Accuracy, EvalMetric, CompositeEvalMetric

def _check_metrics(metrics):
    if isinstance(metrics, CompositeEvalMetric):
        metrics = [m for metric in metrics.metrics for m in _check_metrics(metric)]
    elif isinstance(metrics, EvalMetric):
        metrics = [metrics]
    else:
        metrics = metrics or []
        if not all(isinstance(metric, EvalMetric) for metric in metrics):
            raise ValueError(
                f"metrics must be a Metric or a list of Metric, refer to mxnet.gluon.metric.EvalMetric: {metrics}"
            )
    return metrics

def _check_handler_metric_ref(handler, known_metrics):
    for attribute in dir(handler):
        if any(keyword in attribute for keyword in ['metric' or 'monitor']):
            reference = getattr(handler, attribute)
            if not reference:
                continue
            elif isinstance(reference, list):
                for metric in reference:
                    _check_metric_known(handler, metric, known_metrics)
            else:
                _check_metric_known(handler, reference, known_metrics)

def _check_metric_known(handler, metric, known_metrics):
    if metric not in known_metrics:
        raise ValueError(
            f'Event handler {type(handler).__name__} refers to a metric instance {metric} outside of the known training and validation metrics. Please use the metrics from estimator.train_metrics and estimator.val_metrics instead.'
        )

def _suggest_metric_for_loss(loss):
    return Accuracy() if isinstance(loss, SoftmaxCrossEntropyLoss) else None
