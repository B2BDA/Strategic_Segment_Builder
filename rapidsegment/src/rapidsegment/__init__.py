__version__ = "1.1.3"
__author__ = "Bishwarup Biswas <bishwarup1429@gmail.com>"

from .utils import UniversalDataLoader
from .builder import StrategicSegmentBuilder
from .scorer import StrategicSegmentScore

__all__ = ["UniversalDataLoader", "StrategicSegmentBuilder", "StrategicSegmentScore"]