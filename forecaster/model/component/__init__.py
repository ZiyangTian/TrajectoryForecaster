
from forecaster.model.component import loss
from forecaster.model.component import metrics
from forecaster.model.component import optimizers


def get_metrics(names,
                labels,
                predictions,
                weights=None,
                metrics_collections=None,
                updates_collections=None):
    """ Get metrics by metric names.
        Arguments:
            names: An iterable of `str`s, metric names.
            labels: A `Tensor`.
            predictions: A `Tensor`, must have the same shape with `labels`.
            weights: A `Tensor` broadcastable to the shape of `labels`.
            metrics_collections: A `list` of collections that `metric_value` should be added to.
            updates_collections: A `list` of collections that `update_op` should be added to.
        Returns:
            A pair of `Tensor`s, representing (`metric_value`, `update_op`).
    """
    return dict(map(
        lambda n: (n, getattr(metrics, n)(
            labels,
            predictions,
            weights=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)),
        names))
