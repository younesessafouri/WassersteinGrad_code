from py4castxai.metrics.complexity.sparsness import Sparsness
from  py4castxai.metrics.faithfulness.ROAD import ROAD,ROADCosine
from  py4castxai.metrics.robustness.local_lipschitz_estimate import LocalLipschitzEstimate
from py4castxai.metrics.faithfulness.insertion import InsertionDeletion
registry = {"complexity": {"Sparsness":Sparsness},
            "faithfulness": {"ROAD":ROAD,"ROADCosine": ROADCosine,"InsertionDeletion":InsertionDeletion},
            "robustness": {"LocalLipschitzEstimate": LocalLipschitzEstimate,}
         }