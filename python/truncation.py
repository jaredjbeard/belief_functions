import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from enum import Enum
import numpy as np
from abc import ABC,abstractmethod

class Redistribution(Enum):
    THETA = 0 # Redistributes to Theta, must specify
    LARGEST = 1 # Resdestributes to largest set with mass
    SMALLEST = 2 # Resdestributes to smallest set with mass
    MAXSUBSET = 3 # redistributes to the largest subsets contained in element (if none, defaults to SMALLEST)
    MINSUPERSET = 4 # redistributes to smallest superset containing element (if none, defaults to LARGEST)

class Truncation(ABC):
    """
    Description: A Class for truncating DST belief functions
    """

    @abstractmethod
    def __init__(self, _redist = Redistribution.LARGEST, _theta = None):
        """
        Constructor
        
        Args:
            self (Truncation): object to initialize
            _redist(Redistribution): method for redistributing unassigned mass
            _theta (set(int)): solution space
        Returns:
            Truncation: Truncation object
        """
        super(Truncation, self).__init__()  
        self.redist_ = _redist
        self.theta_ = _theta
        
    @abstractmethod
    def truncate(self):
        """
        truncates distribution
        
        Args:
            self (Truncation): Truncation object
        Returns:
            focal elements, mass
        """
        pass 
    
    def redistribute(self, _inds, _fe, _mass):
        """
        redsitributes truncated mass (see Redistribution enum for details)
        
        Args:
            self (Truncation): Truncation object
            _inds (list(int)): indices to remove
            _fe (list(set(int))): list of focal elements
            _mass (list(int)): list of masses
        Returns:
            focal elements, mass
        """
        l = len(_inds)            
        fe = [None]*l
        m = [None]*l
        
        for i in range(l):
            fe[i] = _fe[_inds[i]]
            m[i] = _mass[_inds[i]]
        
        if self.redist_ == Redistribution.THETA:
            theta_ind = [x for x in _fe if x == self.theta_]
            if not any(theta_ind):
                theta_ind = len(_fe)
                _fe.append(self.theta_)
                _mass.append(0)
            
            _mass[theta_ind] += sum(m)

        elif self.redist_ == Redistribution.LARGEST:
            largest_size = 0
            largest_ind = [0]
            for i in range(len(_fe)):
                if len(_fe[i]) > largest_size:
                    largest_size = len(_fe[i])
                    largest_ind = [i]
                elif largest_size == len(_fe[i]):
                    largest_ind.append(i)
                    
            l = len(largest_ind)
            mass = sum(m)
            for i in range(l):
                _mass[largest_ind[i]] += mass/l
                
        elif self.redist_ == Redistribution.SMALLEST:
            smallest_size = 1e4
            smallest_ind = [0]
            for i in range(len(_fe)):
                if len(_fe[i]) < smallest_size:
                    smallest_size = len(_fe[i])
                    smallest_ind = [i]
                elif smallest_size == len(_fe[i]):
                    smallest_ind.append(i)
                    
            l = len(smallest_ind)
            mass = sum(m)
            for i in range(l):
                _mass[smallest_ind[i]] += mass/l
                 
        elif self.redist_ == Redistribution.MAXSUBSET:

            for i in range(fe):
                largest_size = 0
                largest_ind = [0]
                for j in range(len(_fe)):
                    l = len(_fe[i])
                    if l > largest_size and fe[i].issuperset(_fe[j]):
                        largest_size = len(_fe[j])
                        largest_ind = [j]
                    elif largest_size == len(_fe[j]) and fe[i].issuperset(_fe[j]):
                        largest_ind.append(j)
                    
                l = len(largest_ind)
                if l != 0:
                    for j in range(l):
                        _mass[largest_ind[j]] += m[i]/l
                else:
                    self.redist_ = Redistribution.SMALLEST
                    _fe, _mass = self.redistribute(_inds[i], _fe, _mass)
                    self.redist_ = Redistribution.MAXSUBSET
                
        elif self.redist_ == Redistribution.MINSUPERSET:
            for i in range(fe):
                smallest_size = 0
                smallest_ind = [0]
                for j in range(len(_fe)):
                    l = len(_fe[i])
                    if l < smallest_size and fe[i].issubset(_fe[j]):
                        smallest_size = len(_fe[j])
                        smallest_ind = [j]
                    elif smallest_size == len(_fe[j]) and fe[i].issubset(_fe[j]):
                        smallest_ind.append(j)
                    
                l = len(smallest_ind)
                if l != 0:
                    for j in range(l):
                        _mass[smallest_ind[j]] += m[i]/l
                else:
                    self.redist_ = Redistribution.LARGEST
                    _fe, _mass = self.redistribute(_inds[i], _fe, _mass)
                    self.redist_ = Redistribution.MINSUPERSET
        else:
            print("Invalid Redistribution")
        
        return self.clip(_inds, _fe, _mass)    

    
    def clip(self, _i, _fe, _mass):
        """
        removes mass from distribution
        
        Args:
            self (Truncation): Truncation object
            _i (list(int)): indices to remove
            _fe (list(set(int))): list of focal elements
            _mass (list(int)): list of masses
        Returns:
            focal elements, mass
        """
        _i.sort(reverse=True)
        for el in _i:
            _fe.pop(el)
            _mass.pop(el)
            
        return _fe, _mass
        
        
        
class ThresholdTrunc(Truncation):
    """
    Description: Truncates BF by keeping the elements having mass above a threshold
    
    """

    @abstractmethod
    def __init__(self, _thresh, _is_rel = True, _redist = Redistribution.LARGEST, _theta = None):
        """
        Constructor
        
        Args:
            self (ThresholdTrunc): Object to initialize
            _thresh (float): minimum mass to keep an element
            _is_rel (bool): if false does mass per proposition, otherwise does mass per element of proposition
            _redist(Redistribution): method for redistributing unassigned mass
            _theta (set(int)): solution space
            
        Returns:
            ThresholdTrunc object
        """
        super().__init__(_redist, _theta)  
        
        self.t_ = _thresh
        self.is_rel_ = _is_rel
        
    @abstractmethod
    def truncate(self, _fe, _mass, _thresh = None, _is_rel = None):
        """
        Truncates distribution
        
        Args:
            self (ThresholdTrunc): Truncation object
            _fe (list(set(int))): focal elements
            _mass (list(int)): mass of focal elements
            _thresh (float): minimum mass to keep an element
            _is_rel (bool): if false does mass per proposition, otherwise does mass per element of proposition
            
        Returns:
            focal elements, mass
        """
        if _thresh != None:
            self.t_ = _thresh
        if _is_rel != None:
            self.is_rel_ = _is_rel
        
        inds = []
        fe = []
        for i in range(_fe):
            if self.is_rel_:
                if _mass[i]/len(_fe[i]) < self.t_:
                    inds.append(i)
            else:
                if _mass[i] < self.t_:
                    inds.append(i)
        
        return super.redistribute(inds, _fe, _mass)   

        
class NTermsTrunc(Truncation):
    """
    Description: Truncates BF by keeping the n elements having the most mass
    
    """

    @abstractmethod
    def __init__(self, _n, _is_rel = True, _redist = Redistribution.LARGEST, _theta = None):
        """
        Constructor
        
        Args:
            self (NTermTrunc): Object to initialize
            _n (float): maximum number of terms to keep
            _is_rel (bool): if false does mass per proposition, otherwise does mass per element of proposition
            _redist(Redistribution): method for redistributing unassigned mass
            _theta (set(int)): solution space
            
        Returns:
            NTermsTrun object
        """
        super().__init__(_redist, _theta)  
        
        self.n_ = _n
        self.is_rel_ = _is_rel
        
    @abstractmethod
    def truncate(self, _fe, _mass, _n = None, _is_rel = None):
        """
        Truncates distribution
        
        Args:
            self (NTermTrunc): Truncation object
            _fe (list(set(int))): focal elements
            _mass (list(int)): mass of focal elements
            _n (float): maximum number of terms to keep
            _is_rel (bool): if false does mass per proposition, otherwise does mass per element of proposition
            
        Returns:
            focal elements, mass
        """
        if _n != None:
            self.n_ = _n
        if _is_rel != None:
            self.is_rel_ = _is_rel
        
        inds = []
        l = len(_mass)
        if l > self.n_:
            m = _mass.copy()
            if self.is_rel_:
                for i in range(l):
                    m[i] /= len(_fe[i])
            inds = np.argsort(m)
            return super.redistribute(inds[0:l-self.n_], _fe, _mass)      
        else:
            return _fe, _mass