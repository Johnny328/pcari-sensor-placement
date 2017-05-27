from __future__ import division
from deap import algorithms, base, creator, tools
from joblib import Parallel, delayed
from operator import attrgetter
from tools.network import Network
import tools.objectivefx as obj
import timeit, random, os, copy, numpy, math

NGEN = 1000
MU = 100
LAMBDA = 100
CXPB = 1.0
THRESHOLD = 300

def genetic(filename, network, no_sensors, k, method):
	creator.create("Fitness", base.Fitness, weights=(1, -1, 1))
	creator.create("Chromosome", list, fitness=creator.Fitness)

	def varCustom(population, toolbox, lambda_, cxpb=CXPB):
		offspring = []

		for _ in xrange(lambda_):
			op_choice = random.random()
			if op_choice < cxpb: # produce offspring via crossover
				if method == 'NSGA2': 
					parents = map(toolbox.clone, tools.selTournamentDCD(population, 2))
				elif method == 'SPEA2': 
					parents = map(toolbox.clone, tools.selTournamentSPEA2(population, 2))
				ind1, ind2 = parents[0], parents[1]
				ind1, ind2 = toolbox.mate(ind1, ind2)
				del ind1.fitness.values
				offspring.append(ind1)
			else:
				ind1 = toolbox.clone(random.choice(population))
				#ind1, = toolbox.mutate(ind1)
				del ind1.fitness.values
				offspring.append(ind1)

		for i in range(len(offspring)): # produce offspring via mutation
			offspring[i], = toolbox.mutate(offspring[i])

		return offspring

	def eaCustom(population, toolbox, mu, lambda_, ngen, stats=None, halloffame=None, verbose=__debug__):
		gen = 0
		times = []

		# Evaluate the individuals with an invalid fitness
		logbook = tools.Logbook()
		logbook.header = ['gen', 'nevals', 'avg_time'] + (stats.fields if stats else [])

		# Start timer
		start_time = timeit.default_timer()

		# Evaluate initial (random) population
		invalid_ind = [ind for ind in population if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Record best individuals in hall of fame
		if halloffame is not None:
			halloffame.update(population)

		population[:] = toolbox.select(population, mu)

		# End timer
		elapsed = timeit.default_timer() - start_time
		times.append(elapsed)

		# Record statistics of initial population
		record = stats.compile(population) if stats is not None else {}
		logbook.record(gen=0, nevals=len(invalid_ind), avg_time = elapsed, **record)
		if verbose:
			print logbook.stream

		# Begin the generational process
		gen = gen + 1
		while (gen < ngen):
			#Start timer
			start_time = timeit.default_timer()

			# Vary the population
			offspring = varCustom(population, toolbox, lambda_)

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			# Update the hall of fame with the generated individuals
			if halloffame is not None:
				halloffame.update(offspring)

			#Sort and compute crowding distance
			population[:] = toolbox.select(population + offspring, mu)

			elapsed = timeit.default_timer() - start_time
			times.append(elapsed)
			avg_time = sum(times)/len(times)

			# Update the statistics with the new population
			record = stats.compile(population) if stats is not None else {}
			logbook.record(gen=gen, nevals=len(invalid_ind), avg_time=avg_time, **record)
			if verbose:
				print logbook.stream
			gen = gen + 1

		return gen, avg_time

	locations = network.nodes
	toolbox = base.Toolbox()
	toolbox.register("attr_item", random.sample, locations, no_sensors)
	toolbox.register("chromosome", tools.initRepeat, creator.Chromosome, toolbox.attr_item, n=1)
	toolbox.register("population", tools.initRepeat, list, toolbox.chromosome)
	#toolbox.register("map", mymap)

	def evaluate(chromosome):
		sfpd = network.sfpd
		sensors = []
		for gene in chromosome[0]:
			sensors.append(gene)
		
		dp = obj.coverageFunction(network, sensors)
		sfp = obj.monteCarloSimulation(sensors, sfpd)/len(sensors) 
		fsensors = obj.worstCaseAttack(network, sensors, k)
		reduced_sensors = list(set(sensors) - set(fsensors))
		rdp = obj.coverageFunction(network, reduced_sensors)
		return dp, sfp, rdp

	def crossover(chromosome1, chromosome2): # Uniform Crossover
		if len(set(chromosome1[0])) < no_sensors or len(set(chromosome2[0])) < no_sensors:
			print len(chromosome1[0]), len(chromosome1[0]) # error checking

		parent1 = set(toolbox.clone(chromosome1)[0])
		parent2 = set(toolbox.clone(chromosome2)[0])

		while len(chromosome1[0]) > 0:
			chromosome1[0].pop()
		while len(chromosome2[0]) > 0:
			chromosome2[0].pop()

		intersect = parent1.intersection(parent2)
		for x in intersect:
			chromosome1[0].append(x)
			chromosome2[0].append(x)

		diff1 = list(parent1 - intersect)
		diff2 = list(parent2 - intersect)

		iterr = 0
		while len(chromosome1[0]) < no_sensors:
			rand = random.random()
			if rand <= 0.5:
				chromosome1[0].append(diff1[iterr])
				chromosome2[0].append(diff2[iterr])
			else:
				chromosome1[0].append(diff2[iterr])
				chromosome2[0].append(diff1[iterr])
			iterr = iterr + 1
		return chromosome1, chromosome2

	def mutation(chromosome):
		if len(chromosome[0]) < no_sensors:
			print len(chromosome[0]) # error checking

		# Variable-wise mutation probability
		mutpb = float(1/len(chromosome[0]))

		for i in range(len(chromosome[0])):
			rand = random.random()
			if rand < mutpb:
				chromosome[0].remove(chromosome[0][i])
				gene = random.choice(locations)
				while gene in chromosome[0]:
					gene = random.choice(locations)
				chromosome[0].append(gene)
		return chromosome,

	toolbox.register("evaluate", evaluate)
	toolbox.register("mate", crossover)
	toolbox.register("mutate", mutation)

	if method == 'NSGA2': 
		toolbox.register("select", tools.selNSGA2)
	elif method == 'SPEA2': 
		toolbox.register("select", tools.selSPEA2)

	random.seed(64)
	pop = toolbox.population(n=MU)
	hof = tools.ParetoFront()

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("min", numpy.min, axis=0)
	stats.register("max", numpy.max, axis=0)
	stats.register("avg", numpy.mean, axis=0)
	gen, avg_time = eaCustom(pop, toolbox, mu=MU, lambda_=LAMBDA, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

	return (list(hof), gen, avg_time)