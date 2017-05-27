from __future__ import division
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import tools, base, creator
from tools.network import Network
from utils import Individual, Fitness
import utils
import tools.sfpd as sfpd_
import canvas 
import collections, operator, math, numpy, os, itertools
from colour import Color
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from operator import itemgetter
import collections
import numpy as np
import scipy.linalg
import math
from matplotlib import animation

#Run this script as python -m display.plot
markers = ['o', 'd']
type_ = raw_input("Enter type [1/2/3/3D]: ")

# User preference 'box'
dp_pref = (0.90, 1.0)
risk_pref = (0.0, 0.04)
preferences = [dp_pref, risk_pref]

# Parse results line by line based on the template in results-template
def parseResults(filename, results, method):
	def parseLine(line):
		if line:
			dp, risk, risk_percent, dpa = float(line[0]), float(line[1]), float(line[2]), float(line[3])
			return dp, risk, risk_percent, dpa
		return None

	file = open(filename, 'r')
	line = file.readline()

	while True:
		# Number of sensors
		no_sensors = int(line.strip()) 
		if no_sensors not in results:
			results[no_sensors] = dict()
		results[no_sensors][method] = dict()

		# Attack size
		line = file.readline()
		results[no_sensors][method]["attack-set-size"] = int(line.strip()) 

		# Number of generations
		line = file.readline()
		results[no_sensors][method]["gens"] = int(line.strip()) 

		# Time to run a single generation
		line = file.readline()
		results[no_sensors][method]["single-gen-runtime"] = float(line.strip()) 

		# Time to run the whole simulation
		line = file.readline()
		results[no_sensors][method]['runtime'] = float(line.strip()) 

		# Set of solutions
		line = file.readline()
		results[no_sensors][method]['solution-set'] = dict() 
		while True:
			sensors = frozenset(line.strip().split())
			line = file.readline()
			results[no_sensors][method]['solution-set'][sensors] = parseLine(line.split())
			line = file.readline()
			if line.strip() == '.':
				break

		line = file.readline()
		if not line:
			break

	return results
	
def eliteSet(network, results_dict, methods):
	#Combine all solutions of each method into one pool
	def formPool(no_sensors, results_dict, methods):
		pool = dict()
		for method in methods:
			pool[method] = []
			solutions_set = results_dict[no_sensors][method]['solution-set']
			for solution in solutions_set:
				if type_ == '1':
					fitness = (results_dict[no_sensors][method]['solution-set'][solution][0], # detection performance
							   results_dict[no_sensors][method]['solution-set'][solution][2]) # risk
					weights = (1, -1)
				elif type_ == '2':
					fitness = (results_dict[no_sensors][method]['solution-set'][solution][0], # detection performance
							   results_dict[no_sensors][method]['solution-set'][solution][3]) # dp after attack
					weights = (1, 1)
				elif type_ == '3':
					fitness = (results_dict[no_sensors][method]['solution-set'][solution][3],
							   results_dict[no_sensors][method]['solution-set'][solution][2],) 
					weights = (1, -1)
				elif type_ == '3D':
					fitness = (results_dict[no_sensors][method]['solution-set'][solution][0],
							   results_dict[no_sensors][method]['solution-set'][solution][2],
							   results_dict[no_sensors][method]['solution-set'][solution][3]) 
					weights = (1, -1, 1)
				new_fit = Fitness(fitness, method=method, weights=weights)
				individual = Individual(solution, new_fit)
				pool[method].append(individual)
		return pool

	sfpd = network.sfpd
	for no_sensors in results_dict:
	#for no_sensors in range(20,36):
		print '\nNo. of sensors', no_sensors

		# Create pool of solutions
		pool_inds = formPool(no_sensors, results_dict, methods)
		for method in methods:
			pool_inds[method] = utils.sortNondominated(pool_inds[method])
		pool = [sol for method in methods for sol in pool_inds[method]]
		results_dict[no_sensors]['pool-inds'] = pool_inds
		results_dict[no_sensors]['pool'] = pool
		
		# Extract Pareto front/ elite set
		elite_inds = utils.sortNondominated(pool, first_front_only=True)
		results_dict[no_sensors]['elite-inds'] = elite_inds
		print 'Size of pool: ', str(len(pool)), '\nSize of elite set: ', str(len(elite_inds))

		# Assign tag to each solution
		tags = {method: 0 for method in methods}
		elite_fits = {method: [] for method in methods}
		pool_fits = {method: [] for method in methods}

		for method in methods:
			# Collect fitness values of all solutions in pool
			for sol in pool_inds[method]:
				pool_fits[method].append(sol.fitness.values)

			# Collect fitness values of all solutions in the elite set
			for sol in elite_inds:
				if sol in pool_inds[method]:
					elite_fits[method].append(sol.fitness.values)
					tags[method] += 1
		
		for tag in tags:
			print tag, ': ', tags[tag], '-->', tags[tag]/len(elite_inds), '%'

		results_dict[no_sensors]['pool-fits'] = pool_fits
		results_dict[no_sensors]['elite-fits'] = elite_fits
	
 	return results_dict

def plotResults2D(filepath, no_sensors, results_dict, methods):
	colors = list(i.hex_l for i in Color("blue").range_to(Color("red"), 10))
	#Plot preference box
	def drawPreferenceBox2D(preferences):
		edges_x, edges_y = [], []
		for i in preferences[0]:
			for j in preferences[1]:
				edges_x.append(i)
				edges_y.append(j)
			lines = plt.plot(edges_x, edges_y, '-', color='black')
			edges_x, edges_y = [], []

		edges_x, edges_y = [], []
		for i in preferences[1]:
			for j in preferences[0]:
				edges_x.append(j)
				edges_y.append(i)
			lines = plt.plot(edges_x, edges_y, '-', color='black')
			edges_x, edges_y = [], []

	def plotFigure(no_sensor, fits):
		fig, ax = plt.subplots(figsize=plt.figaspect(0.5)) 
		color_patches = []
		marker_patches = []

		label = 'N=' + str(no_sensors)
		color = colors.pop()

		max_y = 0
		for method, marker in zip(methods, markers):		
			pool_fits = results_dict[no_sensors][fits][method]
			if pool_fits:
				plt.scatter(*zip(*pool_fits), s=30, marker=marker, label=label, facecolor='none', color='blue')

			marker_patches.append(mlines.Line2D([], [], color='blue', marker=marker, fillstyle='none', linestyle = 'None', label=method))
		
		legend1 = plt.legend(handles=marker_patches, loc='upper left')
		plt.gca().add_artist(legend1)

		plt.xlabel('Detection Performance')
		if type_ == '1':
			plt.ylabel('Risk')
		elif type_ == '2':
			plt.ylabel('Detection Performance after Worst-case attack')
		elif type_ == '3':
			plt.ylabel('Risk')
			plt.xlabel('Detection Performance after Worst-case attack')
		
		#drawPreferenceBox2D(preferences)

	def savefig(filepath, fits):
		plotFigure(no_sensors, fits)
		filepath = filepath + '/' + fits
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		plt.savefig(filepath + '/' + str(no_sensors))
	
	savefig(filepath, 'pool-fits')
	savefig(filepath, 'elite-fits')
	plt.close()

def plotResults3D(no_sensors, results_dict, methods):
	def animate(ax, i):
		ax.view_init(elev=10., azim=i)

	colors = list(i.hex_l for i in Color("blue").range_to(Color("red"), 10))
	label = 'N=' + str(no_sensors)
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	plot_color = []
	colors = ['blue', 'orange']
	for method, marker in zip(methods, markers):
		pool_fits = results_dict[no_sensors]['pool-fits'][method]
		color = colors.pop()
		if pool_fits:
			ax.scatter(*zip(*pool_fits), c=color, s=5)
			plot_color.append(color)
	
	scatter1_proxy = lines.Line2D([0],[0], linestyle="none", c=plot_color[0], marker = 'o')
	scatter2_proxy = lines.Line2D([0],[0], linestyle="none", c=plot_color[1], marker = 'o')
	ax.legend([scatter1_proxy, scatter2_proxy], [methods[0], methods[1]], numpoints = 1)

	ax.set_xlabel('Detection Performance')
	ax.set_ylabel('Risk')
	ax.set_zlabel('Reduced Detection Performance')

	for ii in xrange(0,360,1):
		ax.view_init(elev=10., azim=ii)
		plt.savefig("3D-%d.png" % ii)

	plt.show()

def drawGraph(model, network, results_dict, methods):
	sfpd = network.sfpd
	#for no_sensors in results_dict:
	no_sensors = 30
	solutions_set = dict()
	for ind in results_dict[no_sensors]['elite-inds']:
		solutions_set[ind.solution] = ind.fitness.values
	sorted_ = sorted(solutions_set.items(), key=operator.itemgetter(1), reverse=True)
	for i, sensors in enumerate(sorted_):
		print sensors[1]
		graph_path = os.getcwd() + '/results/' + 'Placements/' + model + '/' + str(no_sensors) + '/'
		if not os.path.exists(graph_path):
			os.makedirs(graph_path)
		canvas.drawGraph(network, list(sensors[0]), sensors[1], graph_path + str(i))
	"""
	for method in methods:
		for no_sensors in results_dict:
			sorted_ = sorted(results_dict[no_sensors][method]['solution-set'].items(), key=operator.itemgetter(1), reverse=True)
			for i, sensors in enumerate(sorted_):
				graph_path = os.getcwd() + '/results/' + 'Placements/' + method + '/' + model + '/' + str(no_sensors) + '/'
				if not os.path.exists(graph_path):
					os.makedirs(graph_path)
				canvas.drawGraph(network, sensors[0], sensors[1], graph_path + str(i))
	"""

def main():
	cwd = os.getcwd()
	model = 'ky5'
	methods = ['SPEA2', 'NSGA2']
	filename = cwd + '/models/' + model + '.inp'
	network = Network(filename=filename, max_dist=1000, max_depth=2, option='distance')
	results_dict = dict()
	
	if not network.sfpd:
		network.sfpd = sfpd_.readSFPD(model)

	canvas.drawGraph(network, [], [0,0,0,0], model)

	for method in methods:
		results = 'results/Raw/' + method + '/' + model + '.txt'
		results_dict = parseResults(results, results_dict, method)

	results_dict = eliteSet(network, results_dict, methods)

	if type_ == '3D':
		#plotResults3D(30, results_dict, methods)
		drawGraph(model, network, results_dict, methods)
	else:		
		no_sensors_list = [x for x in results_dict]
		for no_sensors in no_sensors_list:
			filepath = 'results/' + '/Tradeoffs' + '/' + str(type_) + '/' 
			plotResults2D(filepath, no_sensors, results_dict, methods)
	
if __name__ == '__main__':
    main()
