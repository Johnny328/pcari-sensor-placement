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
import pandas
import pylab as pl
import imageio

#Run this script as python -m display.plot
methods = ['NSGA2', 'SPEA2']
markers = ['o', 'd']

# User preference 'box'
dp_pref = (0.90, 1.0)
risk_pref = (0.0, 0.04)
preferences = [dp_pref, risk_pref]

# Parse results line by line based on the template in results-template
def parseResults(model, results):
	def cleanup(results_dict):
		to_delete = []
		for no_sensors in results_dict:
			if methods[0] in results_dict[no_sensors] and methods[1] not in results_dict[no_sensors]:
				to_delete.append(no_sensors)
			elif methods[1] in results_dict[no_sensors] and methods[0] not in results_dict[no_sensors]:
				to_delete.append(no_sensors)
		for no_sensors in to_delete:
			del results_dict[no_sensors]

	def parseLine(line):
		if line:
			dp, risk, risk_percent, dpa = float(line[0]), float(line[1]), float(line[2]), float(line[3])
			return dp, risk, risk_percent, dpa
		return None

	for method in methods:
		file = open('results/Raw/' + method + '/' + model + '-attack-' + method +'.txt', 'r')
		line = file.readline()
		while True:
			# Number of sensors
			no_sensors = int(line.strip()) 
			
			# Attack size
			attack_size = file.readline()
			no_sensors = int(attack_size)
			print no_sensors
			if no_sensors not in results:
				results[no_sensors] = dict()

			results[no_sensors][method] = dict()
			results[no_sensors][method]["attack-set-size"] = int(attack_size.strip()) 

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
	cleanup(results)
	return results
	
def eliteSet(network, results_dict, type_, verbose=False):
	#Combine all solutions of each method into one pool
	def formPool(no_sensors, results_dict, methods, type_):
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
	for no_sensors in results_dict: #[20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33]:
		if verbose: print '\nNo. of sensors', no_sensors

		# Create pool of solutions
		pool_inds = formPool(no_sensors, results_dict, methods, type_)
		for method in methods:
			pool_inds[method] = utils.sortNondominated(pool_inds[method])
		pool = [sol for method in methods for sol in pool_inds[method]]
		results_dict[no_sensors]['pool-inds'] = pool_inds
		results_dict[no_sensors]['pool'] = pool
		
		# Extract Pareto front/ elite set
		elite_inds = utils.sortNondominated(pool, first_front_only=True)
		results_dict[no_sensors]['elite-inds'] = elite_inds
		if verbose: print 'Size of pool: ', str(len(pool)), '\nSize of elite set: ', str(len(elite_inds))

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
		if verbose:
			for tag in tags:
				print tag, ': ', tags[tag], '-->', tags[tag]/len(elite_inds), '%'

		results_dict[no_sensors]['pool-fits'] = pool_fits
		results_dict[no_sensors]['elite-fits'] = elite_fits
	
 	return results_dict

def plotResults2D(filepath, no_sensors, results_dict, type_):
	#colors = list(i.hex_l for i in Color("blue").range_to(Color("red"), 3))
	colors = ['blue', 'red']

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

	def plotFigure(no_sensor, fits, type_):
		fig, ax = plt.subplots(figsize=plt.figaspect(0.5)) 
		marker_size = 20
		color_patches = []
		marker_patches = []

		label = 'N=' + str(no_sensors)

		max_y = 0
		n = 0
		#for method, marker in zip(methods, markers):
		marker = 'o'
		for method in methods:
			#color = colors.pop()
			color = colors[n]
			if fits in results_dict[no_sensors]:
				pool_fits = results_dict[no_sensors][fits][method]
				if pool_fits: plt.scatter(*zip(*pool_fits), s=marker_size, marker=marker, label=label, color=color)
			marker_patches.append(mlines.Line2D([], [], color=color, marker=marker, fillstyle='full', linestyle = 'None', label=method))
			n += 1
		
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

	def savefig(filepath, fits, type_):
		plotFigure(no_sensors, fits, type_)
		filepath = filepath + '/' + fits
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		plt.savefig(filepath + '/' + str(no_sensors))
	
	savefig(filepath, 'pool-fits', type_)
	#savefig(filepath, 'elite-fits', type_)
	plt.close()

def plotResults3DbyN(filepath, no_sensors_, results_dict):
	def animate(ax, i):
		ax.view_init(elev=10., azim=i)

	#colors = ['red', 'violet', 'blue']
	colors = list(i.hex_l for i in Color("darkblue").range_to(Color("red"), 10))
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	plot_color = []
	method = 'NSGA2'
	no_sensors_ = sorted(no_sensors_, reverse=False)
	percent = [r"k=" + str(x) +" (" + str(x/20) + r"n)" for x in no_sensors_]
	for no_sensors in no_sensors_:
		pool_fits = results_dict[no_sensors]['pool-fits'][method]
		color = colors.pop()
		if pool_fits:
			ax.scatter(*zip(*pool_fits), c=color, s=5)
			plot_color.append(color)
	
	scatter_proxy = []
	for index, i in enumerate(no_sensors_):
		scatter_proxy.append(lines.Line2D([0],[0], linestyle="none", c=plot_color[index], marker = 'o'))
	ax.legend(scatter_proxy, percent, numpoints = 1)

	ax.set_xlabel('Detection Performance')
	ax.set_ylabel('Risk')
	ax.set_zlabel('Reduced Detection Performance')
	#plt.show()
	
	for ii in xrange(0,360,1):
		ax.view_init(elev=10., azim=ii)
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		filename = filepath+"%d.png" % ii
		plt.savefig(filename)

	plt.close()

def plotResults3D(filepath, no_sensors, results_dict):
	def savefig(path, fits):
		path = path+'/'+fits+"/"
		for ii in xrange(0,360,1):
			ax.view_init(elev=10., azim=ii)
			if not os.path.exists(path):
				os.makedirs(path)
			filename = path+"%d.png" % ii
			plt.savefig(filename)

	def animate(ax, i):
		ax.view_init(elev=10., azim=i)

	for fits in ['elite-fits']:
		colors = list(i.hex_l for i in Color("blue").range_to(Color("red"), 10))
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		plot_color = []
		colors = ['blue', 'orange']
		for method, marker in zip(methods, markers):
			pool_fits = results_dict[no_sensors][fits][method]
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

		savefig(filepath, fits)
		plt.close()

def drawGraph(no_sensors, model, network, results_dict):
	sfpd = network.sfpd
	solutions_set = dict()
	for ind in results_dict[no_sensors]['elite-inds']:
		solutions_set[ind.solution] = ind.fitness.values
	sorted_ = sorted(solutions_set.items(), key=operator.itemgetter(1), reverse=True)
	for i, sensors in enumerate(sorted_):
		graph_path = os.getcwd() + '/results/' + 'Placements/' + model + '/' + str(no_sensors) + '/'
		if not os.path.exists(graph_path):
			os.makedirs(graph_path)
		canvas.drawGraph(network, list(sensors[0]), sensors[1], graph_path + str(i))

def boxplot(filepath, no_sensors, results_dict):
	fig, axes = plt.subplots(nrows=2, ncols=3)
	fig.subplots_adjust(hspace=.5)
	fig.tight_layout()
	n = -1

	for method in methods:
		n += 1
		pool_fits = results_dict[no_sensors]['pool-fits'][method]
		data = []
		for fits in pool_fits:
			tuple_ = {'Detection Performance':fits[0], 'Risk':fits[1], 'Reduced Detection Performance':fits[2]}
			data.append(tuple_)
		df = pandas.DataFrame(data)
		axes[n, 1].set_title(method)
		ax = df.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, ax=axes[n])
	
		print no_sensors
		print df.describe()

	for i in range(0,3):
		min_ = min(axes[n,i].get_ylim()[0], axes[n-1,i].get_ylim()[0])
		max_ =   max(axes[n,i].get_ylim()[1], axes[n-1,i].get_ylim()[1])
		axes[n,i].set_ylim(min_,max_)
		axes[n-1,i].set_ylim(min_,max_)
	
	if not os.path.exists(filepath):
		os.makedirs(filepath)
	plt.savefig(filepath+"%d.png" %no_sensors)

def drawplots(network, results_dict, filename):
	#canvas.drawGraph(network, [], [0,0,0,0], model)
	results_dict_ = parseResults(model, results_dict)
	types = ['1', '2', '3', '3D']
	for type_ in types:
		print "Processing " + type_ + '...'
		if type_ == '3D': verbose = True
		else: verbose = False 
		results_dict = eliteSet(network, results_dict_, type_, verbose)
		if type_ == '3D': 
			filepath = 'results' + '/3D' + '/' + model + '/'
			for no_sensors in [30]:
				print "Creating boxplot for ", no_sensors
				boxplot('results' + '/Boxplot' + '/' + model + '/', no_sensors, results_dict)
				print "Creating 3D plot for ", no_sensors
				plotResults3D(filepath + str(no_sensors) + '/', no_sensors, results_dict)
				print "Drawing graph for ", no_sensors
				drawGraph(no_sensors, model, network, results_dict)
			plotResults3DbyN(filepath + '/n/', [x for x in results_dict], results_dict)
		else: 
			for no_sensors in results_dict:
				filepath = 'results/' + '/Tradeoffs' + '/' + model + '/' + str(type_) + '/' 
				plotResults2D(filepath, no_sensors, results_dict, type_)

def main():
	cwd = os.getcwd()
	models = ['makati']

	for model in models:
		filename = cwd + '/models/' + model + '.inp'
		network = Network(model=model, max_dist=1000, max_depth=2, option='distance')
		results_dict = dict()
		if not network.sfpd:
			network.sfpd = sfpd_.readSFPD(model)

		
if __name__ == '__main__':
    main()
