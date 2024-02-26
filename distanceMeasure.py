from abc import abstractmethod, ABC
import numpy as np

"""
    Definition of an abstract class DistanceMeasure so that it is possible 
    generalize the use of algorithms accordig to different metrics
"""
class DistanceMeasure(ABC):
    
    @abstractmethod
    def compute_distance(self, x, y):
        pass    

"""
    EuclideanDistance class definition
"""
class EuclidianDistance(DistanceMeasure):

    def compute_distance(self, x, y):
        assert len(x) == len(y)
        d = np.array([x[i] - y[i] for i in range(len(x))])
        d = np.power(d, 2)
        d = np.sum(d)
        d = np.power(d, 1/2)
        return d
    
"""
    ManhattanDistance class definition
"""
class ManhattanDistance(DistanceMeasure):
    def compute_distance(self, x, y):
        assert len(x) == len(y)
        d = np.array([abs(x[i] - y[i]) for i in range(len(x))])
        d = np.sum(d)
        return d
    
"""
    JaccardDistance class definition
    1 max distance, no elements in common, 0 all elements in common
"""
class JaccardDistance(DistanceMeasure):
    def compute_distance(self, x, y):
        assert len(x)==len(y)
        
        common = 0
        for i in range(len(x)):
            if x[i] == y[i] and x[i] != 0:
                common += 1
        print(common)
        d = 1 - common/len(x)
        return d
    
"""
    HammingDistance class definition
"""
class HammingDistance(DistanceMeasure):
    def compute_distance(self, x, y):
        assert len(x)==len(y)
        common = np.sum(np.array(x) == np.array(y))
        d = 1 - common/len(x)
        return d