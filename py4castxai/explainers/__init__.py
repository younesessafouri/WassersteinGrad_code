from .smoothGrad import SmoothGrad, StatGrad
from .inputxgrad import InputxGrad
from .basegrad import BaseGrad
from .XGrad import XGrad
from .smoothGradSquared import SmoothGradSquared
from .WassersteinGrad import (
    WassersteinGrad,
    WassersteinSmoothGrad,
    WassersteinGradMask,
)
from .otGrad import otGrad
from .Centroid import (
    Centroid,
    CentroidSmoothGrad,
    CentroidWassersteinGrad,
)
from .VarGrad import VarGrad
from .integratedGrad import IntegratedGrad


__all__ = [
    "SmoothGrad",
    "SmoothGradSquared",
    "InputxGrad",
    "BaseGrad",
    "XGrad",
    "WassersteinGrad",
    "WassersteinSmoothGrad",
    "WassersteinGradMask",
    "otGrad",
    "Centroid",
    "CentroidSmoothGrad",
    "CentroidWassersteinGrad",
    "VarGrad",
    "IntegratedGrad",
    "StatGrad",
]


explainers_registry = {
    "SmoothGrad": SmoothGrad,
    "BaseGrad": BaseGrad,
    "SmoothGradSquared": SmoothGradSquared,
    "InputxGrad": InputxGrad,
    "XGrad": XGrad,

    "WassersteinGrad": WassersteinGrad,
    "WassersteinSmoothGrad": WassersteinSmoothGrad,
    "WassersteinGradMask": WassersteinGradMask,

    "otGrad": otGrad,

    "Centroid": Centroid,
    "CentroidSmoothGrad": CentroidSmoothGrad,
    "CentroidWassersteinGrad": CentroidWassersteinGrad,

    "VarGrad": VarGrad,
    "IntegratedGrad": IntegratedGrad,
    "StatGrad": StatGrad,
}