from .base import Explainer


class BaseGrad(Explainer):
    """Plain input gradient (Baehrens et al., 2010; Simonyan et al., 2013)."""

    def explain(self, batch, ctx):
        return self.gradient(batch, ctx)
