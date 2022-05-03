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
        self.size_theta_ = len(self.theta_i_)
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
        
        if _truncation == None:
            self.trunc_ = NTermsTrunc(1.5e4)
        else:
            self.trunc_ = _truncation
            
        self.rng_ = np.random.default_rng()
     
    ###########################################################
    # Translation
    ###########################################################   
    def translate_source_2_int(self, _fe):
        """
        Translate aribitrary focal elements to integer based ones to simplify internal logic

        Args:
            self (BeliefFunction): Belief function
            _fe (list(object)): list of focal elements
        Returns:
            list(int)
        """
        fe = []
        for proposition in _fe:
            t = {}
            for el in range(proposition):
                t.add(self.mapping_[el])
    
    def translate_int_2_source(self, _fe):
        """
        Translate local focal elements to specified type

        Args:
            self (BeliefFunction): Belief function
            _fe (list(object)): list of focal elements
        Returns:
            list(object)
        """
        fe = []
        for proposition in _fe:
            t = {}
            for el in range(proposition):
                t.add(self.mapping_.get(el))
    
    def translate_pair_2_list(self, _dist):
        """
        Convert list of (element, mass) pairs to lists

        Args:
            self (BeliefFunction): Belief function
            _dist (list(tuple)): distribution represented as (focal elements, mass)
        Returns:
            focal elements, mass
        """
        return list(zip(*_dist[0])), list(zip(*_dist[1]))
    
    def translate_list_2_pair(self, _fe, _mass):
        """
        Convert lists to (element, mass) pairs

        Args:
            self (BeliefFunction): Belief function
            _fe (list(set)): focal elements
            _mass (list(int)): mass of focal elements
        Returns:
            distribution
        """
        dist = [None]*len(_mass)
        i = 0
        for el,m in zip(_fe,_mass):
            dist[i] = (el,m)
            i += 1
        return dist
    
    ###########################################################
    # Combination
    ###########################################################              
    def dempster_combination_bf(self, _bf):
        """
        Combine belief function with that supplied

        Args:
            self (BeliefFunction): Belief function
            _bf (BeliefFunctin): belief function to combine
        Returns:
            dist, conflict
        """
        return self.dempster_combination(_bf.translate_int_2_source(_bf.fe_), _bf.mass_)
                
    def dempster_combination_dist(self, _dist):
        """
        Combine belief function with a BPA described by focal elements and mass

        Args:
            self (BeliefFunction): Belief function
            _dist (list(tuple)): distribution represented as (focal elements, mass)
        Returns:
            dist, conflict
        """
        _fe, _mass = self.translate_pair_2_list(_dist)
        
        return self.dempster_combination(_fe,_mass)
           
    def dempster_combination(self, _fe, _mass):
        """
        Combine belief function with a BPA described by focal elements and mass

        Args:
            self (BeliefFunction): Belief function
            _fe (list(set(int))): focal elements
            _mass (list(int)): mass of focal elements
        Returns:
            dist, conflict
        """
        fe = self.translate_source_2_int(_fe)

        temp_fe = []
        temp_mass = []
        conflict = 0.0
        for i in range(len(self.mass_)):
            for j in range(len(_mass)):
                
                el_intersect = self.fe_[i].intersection(fe[j])

                if bool(el_intersect):
                    if el_intersect in temp_fe:
                        temp_mass[temp_fe.index(el_intersect)] += self.mass_[i]*_mass[j]
                    else:
                        temp_fe.append(el_intersect)
                        temp_mass.append(self.mass_[i]*_mass[j])
                else:
                    conflict += self.mass_[i]*_mass[j]

        if conflict != 1:
            temp_mass = [x/(1-conflict) for x in temp_mass]  
            self.conflict = conflict
            self.fe_, self.mass_ = self.trunc_.truncate(self.fe_, self.mass_)
        else:
            print("incompatible beliefs ", _fe, "|", self.fe_)
            
        return self.translate_list_2_pair(self.translate_int_2_source(self.fe_),self.mass__), self.conflict

    ###########################################################
    # Belief
    ########################################################### 
    def get_belief_function(self):
        """
        Computes belief function
        
        Args:
            self (BeliefFunction): Belief function
        Returns:
            belief
        """
        b = [0] * self.size_theta_
        i = 0
        for el in self.theta:
            b[i] = self.compute_belief({el})
            i += 1
            
        return self.translate_int_2_source(list(range(self.size_theta_)),b)
    
    def compute_belief(self, A):
        """
        Computes belief of A
        
        Args:
            self (BeliefFunction): Belief function
            A (set(int)): porposition to compute belief of
        Returns:
            belief
        """
        b = 0
        for el, m in zip(self.fe_, self.mass_):
            if bool(A.issuperset(el)):
                b += m
        return b
    
    ###########################################################
    # Plausibility
    ###########################################################   
    def get_plausibility_function(self):
        """
        Computes plausibility function
        
        Args:
            self (BeliefFunction): Belief function
        Returns:
            plasuibility
        """
        pl = [0] * self.size_theta_
        i = 0
        for el in self.theta:
            pl.append(self.compute_plausibility({el}))
            i += 1
        return self.translate_int_2_source(list(range(self.size_theta_)),pl)
    
    def compute_plausibility(self, A):
        """
        Computes plausibility of A
        
        Args:
            self (BeliefFunction): Belief function
            A (set(int)): proposition to compute plausibility of
        Returns:
            plausibility
        """
        pl = 0
        for el, m in zip(self.fe_, self.mass_):
            if bool(el.intersection(A)):
                pl += m
        return pl
    
    ###########################################################
    # BPA
    ###########################################################  
    def get_bpa(self):
        """
        returns basic probability assignment
        
        Args:
            self (BeliefFunction): Belief function
        Returns:
            BPA
        """
        return self.translate_int_2_source(self.translate_int_2_source(self.fe_), self.mass_)
    
    ###########################################################
    # Pignistic
    ###########################################################  
    def get_pignistic_prob(self):
        """
        Computes pignistic probability distribution
        
        Args:
            self (BeliefFunction): Belief function
        Returns:
            distribution
        """
        p = [0] * self.size_theta_
        i = 0
        for el in self.theta:
            p.append(self.compute_pignistic_prob({el}))
            i += 1
        return self.translate_int_2_source(list(range(self.size_theta_)),p)
    
    def _compute_pignistic_prob(self,A):
        """
        Computes pignistic probability of A
        
        Args:
            self (BeliefFunction): Belief function
            A (set(int)): element to compute plausibility of
        Returns:
            pignistic probability
        """
        p = 0
        for el, m in zip(self.fe_, self.mass_):
            if bool(el.issuperset(A)):
                p += m/len(el)
        return p
    
    ###########################################################
    # Sampling
    ###########################################################
    def _sample_probability(self, _only_nonzero = False):

        prob = [0] * self.size_theta_
        
        for i in range(len(self.mass_)):
            
            p = np.zeros([self.size_theta_,1])
            for j in self.fe_[i]:
                
                if (not _only_nonzero) or (_only_nonzero and self._compute_belief({j})):
                    p[j][0] = self.rng_.uniform()
                else:
                    p[j][0] = 0
            p_sum = np.sum(p)
            for j in self.fe_[i]:
                prob[j] += p[j][0]*self.mass_[i]/p_sum
        return self.translate_int_2_source(list(range(self.size_theta_)),prob)
    
    ###########################################################
    # Conflict
    ###########################################################  
    def _get_weight_of_conflict(self):
        """
        Computes weight of conflict
        
        Args:
            self (BeliefFunction): Belief function
        Returns:
            weight of conflict
        """
        return math.log(1/(1-self.conflict))
    #################################


