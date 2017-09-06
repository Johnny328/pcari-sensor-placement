from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, copy, math, timeit, random, os, itertools
from datetime import datetime, timedelta
from deap import base, creator
from tools.network import Network
import tools.sfpd as sfpd_
import tools.objectivefx as obj
import emo
from random import shuffle
from networkx import *
import display.canvas as canvas

""" Get running time of each algorithm """
def _template_func(setup, func):
	def inner(_it, _timer, _func=func):
		setup()
		_t0 = _timer()
		for _i in _it:
			retval = _func()
		_t1 = _timer()
		return _t1 - _t0, retval
	return inner
timeit._template_func = _template_func

def genetic_(filename, network, no_sensors, max_fail, method):
	t = timeit.Timer(lambda: emo.genetic(filename, network, no_sensors, max_fail, method))# pareto optimal curve
	time_ga, sensors  = t.timeit(number=1)
	ga_sensors_, gen, avg_time = sensors[0], sensors[1], sensors[2]

	# Write test run info onto file
	with open(filename, 'a') as result:
		result.write(str(no_sensors) + "\n" + \
			str(max_fail) + '\n' + \
			str(gen) + '\n' + \
			str(time_ga)+ '\n' + \
			str(avg_time) + "\n"
			)

	# Write sensor info and performance onto file
	for sensors in ga_sensors_:
		ga_sensors = list(sensors[0])

		dp = obj.coverageFunction(network, ga_sensors)
		risk = obj.monteCarloSimulation(ga_sensors, network.sfpd)
		ga_attack = obj.worstCaseAttack(network, ga_sensors, max_fail)
		rdp = obj.coverageFunction(network, list(set(ga_sensors) - set(ga_attack)))

		# Write results to file
		with open(filename, 'a') as result:
			string = ''
			for sensor in ga_sensors:
				string = string + sensor + " "
			result.write(string +'\n' + \
				str(dp) + ' ' + \
				str(risk) + " " + \
				str(float(risk/no_sensors)) + ' ' + \
				str(rdp) +'\n'
			)

	# Mark the end of results
	with open(filename, 'a') as result:
			result.write('.\n')

def MonteCarloSimulation(network):
	sfpd = network.sfpd
	diff = 0
	no = 30

	for k in range(1, no + 1):
		i = 1
		rsensors = random.sample(network.graph.nodes(), i)
		estimate = obj.monteCarloSimulation(rsensors, sfpd)
		true =  obj.sensorFailureRisk(rsensors, sfpd)
		diff = diff + abs(obj.monteCarloSimulation(rsensors, sfpd) - obj.sensorFailureRisk(rsensors, sfpd))
		print len(rsensors), estimate, true, diff
	diff = diff/no 
	print diff

def main():
	cwd = os.getcwd()
	model = 'makati'
	methods = ['NSGA2', 'SPEA2']
	
	percent_attacked = 0.3
	sfpd_bound = 0.3
	start, finish = 30, 50

	option = 'distance'
	max_dist = 1000
	max_depth = 2

	print "Initializing graph..."
	network = Network(model=model, max_dist=max_dist, max_depth=max_depth, sfpd_bound=sfpd_bound, option=option)
	print "No. of nodes: ", len(network.nodes), " No. of edges", len(network.edges)
	print "Length (km):", sum(edge[2]['length'] for edge in network.graph.edges(data=True))/(1000)
	print network.sfpd

	for no_sensors in range(start, finish+1):
		for method in methods:
			filename = 'results/Raw/' + method + '/' + model + '-' + method +'.txt'
			max_fail = int(math.ceil(no_sensors*percent_attacked))
			genetic_(filename, network, no_sensors, max_fail, method)

if __name__ == '__main__':
    main()