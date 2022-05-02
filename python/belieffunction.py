import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from operator import truediv
import random
import math
import numpy as np

from truncation import NTermsTrunc

class BeliefFunction():
    """
    Description: A Class for representing Dempster-Shafer Belief Functions
    User defines:
        _solution_space (list): list of elements representing distribution outcomes
        _truncation (Truncation): Technqiue for truncatin distribution
    """



    def __init__(self, _solution_space, _truncation = None, _fe = None, _mass = None):
        """
        Constructor

        Args:
            self (BeliefFunction): Object to initialize
            _solution_space (list): list of elements (must be hashable) representing distribution outcomes
            _truncation (Truncation): Technqiue for truncatin distribution
            _fe (list(set(object))) = prior set of focal elements
            _mass (list(float)) = prior mass assignment to focal elements
        Returns:
            BeliefFunction: BF object
        """
        super(BeliefFunction, self).__init__()    
        self.theta_ = _solution_space
        self.theta_i_ = list(range(len(_solution_space)))
        self.mapping_ = {}
        for i in range(len(self.theta_)):
            self.mapping_[self.theta_[i]].append(self.theta_i_[i])	
        
        if _fe == None:
            self.fe_ = [set(self.theta_i_)]
        else:
            self.fe_ = _fe
        
        if _mass == None:
            self.mass_ = [1]
        else:
            self.mass_ = _mass
        # self.plausibility = [1]
        self.conflict_ = 0
        
        self.trunc_ = _truncation
            
        self.rng_ = np.random.default_rng()

##############################################################
    def _dempster_combination(self, _focal_elements, _mass):
        fe = []
        for el in _focal_elements:
            fe.append(set(el))
        #fe = set(frozenset(el) for el in fe)
        temp_focal_elements = []
        temp_mass = []
        conflict = 0.0
        for i in range(len(self.mass)):
            
            for j in range(len(_mass)):
                el_intersect = self.focal_elements[i].intersection(fe[j])
                # print("-------------------------------")
                # print("fe1: ", self.focal_elements[i])
                # print("fe2: ", fe[j])
                # print("Intersect", el_intersect)
                if bool(el_intersect):
                    if el_intersect in temp_focal_elements: #check if this works on sets.....
                        temp_mass[temp_focal_elements.index(el_intersect)] += self.mass[i]*_mass[j]
                    else:
                        temp_focal_elements.append(el_intersect)
                        temp_mass.append(self.mass[i]*_mass[j])
                else:
                    conflict += self.mass[i]*_mass[j]
        # print("Temp: ", temp_focal_elements)

        if conflict != 1:
            temp_mass = [x/(1-conflict) for x in temp_mass]  
            self.conflict = conflict
            self.focal_elements, self.mass = self.trunc_.truncate(self.focal_elements, self.mass)
        else:
            print("incompatible beliefs ", _focal_elements, "|", self.focal_elements)

    def _compute_belief(self, A):
        b = 0
        for el, m in zip(self.focal_elements, self.mass):
            if bool(A.issuperset(el)):
                b += m
        return b


    def _compute_plausibility(self, A):
        pl = 0
        for el, m in zip(self.focal_elements, self.mass):
            if bool(el.intersection(A)):
                pl += m
        return pl

    # def _sample_distribution(self)

    def _compute_pignistic_prob(self,A):
        b = 0
        for el, m in zip(self.focal_elements, self.mass):
            if bool(el.issuperset(A)):
                b += m/len(el)
        return b
    
    def _get_element_beliefs(self):
        b = []
        for el in self.theta:
            b.append(self._compute_belief({el}))
        return b

    def _get_pignistic_prob(self):
        b = []
        for el in self.theta:
            b.append(self._compute_pignistic_prob({el}))
        return b
    
    def _get_element_plaus(self):
        b = []
        for el in self.theta:
            b.append(self._compute_plausibility({el}))
        return b 

    def _get_weight_of_conflict(self):
        return math.log(1/(1-self.conflict))
    
    def _sample_probability(self):
        l = len(self.theta)
        b = [0] * l
        for i in range(len(self.mass)):
            p = np.zeros([l,1])
            for j in self.focal_elements[i]:
                p[j][0] = self.rng_.uniform()
            p_sum = np.sum(p)
            for j in self.focal_elements[i]:
                b[j] += p[j][0]*self.mass[i]/p_sum
        return b
    
    def _sample_nonzero_probability(self):
        l = len(self.theta)
        b = [0] * l
        
        for i in range(len(self.mass)):
            p = np.zeros([l,1])
            for j in self.focal_elements[i]:
                # print(j, " | ",self._compute_belief({j}))
                if self._compute_belief({j}):
                    p[j][0] = self.rng_.uniform()
                else:
                    p[j][0] = 0
            p_sum = np.sum(p)
            for j in self.focal_elements[i]:
                b[j] += p[j][0]*self.mass[i]/p_sum
        return b      

