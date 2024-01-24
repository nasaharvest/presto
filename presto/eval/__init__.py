from .algae_blooms_eval import AlgaeBloomsEval
from .cropharvest_eval import CropHarvestEval, CropHarvestMultiClassValidation
from .croptype_france_eval import CroptypeFranceEval
from .eurosat_eval import EuroSatEval
from .eval import EvalTask
from .fuel_moisture_eval import FuelMoistureEval
from .treesat_eval import TreeSatEval

__all__ = [
    "CropHarvestEval",
    "CropHarvestMultiClassValidation",
    "CroptypeFranceEval",
    "EvalTask",
    "EuroSatEval",
    "FuelMoistureEval",
    "AlgaeBloomsEval",
    "TreeSatEval",
]
