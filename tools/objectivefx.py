from __future__ import division
from math import *
import itertools
import random
import ujson

""" Running time of each evaluation: O(sigma*n)
	where sigma = no. of iterations
		  n = number of sensors
"""
def monteCarloSimulation(sensors, sfpd):
	random.seed(64)
	no_sim = 100000
	total_no_faults = 0
	for i in range(no_sim):
		no_faults = 0
		for sensor in sensors:
			rand = random.random()
			if rand <= sfpd[sensor]:
				no_faults = no_faults + 1
		total_no_faults = total_no_faults + no_faults
	return total_no_faults/no_sim

""" Running time of each evaluation: O(2^n)
	where n = number of sensors
"""
def sensorFailureRisk(sensors, sfpd):
	def getFailureProb(sensors, sfpd, k): # Accounts for scenarios where k sensors fail
		sum_ = 0
		for fsubset in itertools.combinations(sensors, k):
			product = 1
			for sensor in sensors:
				if sensor in fsubset:
					product = product*sfpd[sensor]
				else:
					product = product*(1-sfpd[sensor])
			sum_ = sum_ + product
		return sum_

	sum_total = 0
	for i in range(0, len(sensors) + 1):
		sum_total = sum_total + i*getFailureProb(sensors, sfpd, i)
	return sum_total

def coverageFunction(network, sensors):
	total_coverage = []
	for sensor in sensors:
		total_coverage = list(set(total_coverage).union(set(network.range_list[sensor])))
	return float(len(total_coverage)/len(network.edges))

def worstCaseAttack(network, sensors, k):
	best_attack = []

	while len(best_attack) < k:
		dp_min = 10000
		sensor_attack = None

		for sensor in list(set(sensors) - set(best_attack)):
			temp_attack = ujson.loads(ujson.dumps(best_attack))
			temp_attack.append(sensor)
			reduced_sensors = list(set(sensors) - set(temp_attack))
			dp = coverageFunction(network, reduced_sensors)
			
			if dp < dp_min:
				dp_min = dp
				sensor_attack = sensor

		best_attack.append(sensor_attack)
	return best_attack
