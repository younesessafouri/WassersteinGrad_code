from .base import Explainer
from .basegrad import BaseGrad
from .integratedgrad import IntegratedGrad
from .smoothgrad import SmoothGrad
from .vargrad import VarGrad
from .wassersteingrad import WassersteinGrad, WassersteinGradMask, wasserstein_barycenter

explainers_registry = {
    "BaseGrad": BaseGrad,
    "SmoothGrad": SmoothGrad,
    "VarGrad": VarGrad,
    "IntegratedGrad": IntegratedGrad,
    "WassersteinGrad": WassersteinGrad,  # WGBary
    "WassersteinGradMask": WassersteinGradMask,  # WGBary x Grad
}

__all__ = [
    "Explainer",
    "BaseGrad",
    "SmoothGrad",
    "VarGrad",
    "IntegratedGrad",
    "WassersteinGrad",
    "WassersteinGradMask",
    "wasserstein_barycenter",
    "explainers_registry",
]
