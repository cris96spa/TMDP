"""
    Definition of an abstract class DistanceMeasure so that it is possible 
    generalize the use of algorithms accordig to different metrics
"""

from abc import abstractmethod, ABC


class DistanceMeasure(ABC):
    
    @abstractmethod
    def compute_distance(x, y):
        pass    
