from operator import mul, truediv
from tools.network import Network
import tools.objectivefx as obj
from deap import tools, base, creator
from collections import defaultdict
import copy, math, random
import ujson

class Individual:
	def __init__(self, solution, fitness):
		self.solution = frozenset(solution)
		self.fitness = fitness

class Fitness:
	def __init__(self, values, method, weights=(1,1), tolerance = 0.0005):
		self.weights = weights
		self.tolerance = tolerance
		self.setValues(values)

	def setValues(self, values):
		self.wvalues = tuple(map(mul, values, self.weights))
		self.values = values 

	def dominates(self, other, obj=slice(None)):
		dom = False
		for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
			if self_wvalue > other_wvalue:
				dom = True
			elif self_wvalue < other_wvalue:
				return False
		return dom

def sortNondominated(individuals, first_front_only=True):
    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = map_fit_ind.keys()
    
    current_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)
	    
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)
	    
    front = []
    for fit in current_front:
        front.append(map_fit_ind[fit][0])
    return front
