def normalize_attribution(a):
    """|a| / max|a|, applied to every attribution map before scoring."""
    return a.abs() / (a.abs().max() + 1e-8)


class Metric:
    """Base class of the evaluation metrics.

    ``evaluate(events, ctx)`` runs the explainer on every event, an iterable of
    ``(batch_idx, runtime, batch)``, and returns ``{score_name: [one score per event]}``.
    """

    def __init__(self, model, explainer, seed=42):
        self.model = model
        self.explainer = explainer
        self.seed = seed

    def evaluate(self, events, ctx):
        raise NotImplementedError
