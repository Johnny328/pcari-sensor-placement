from __future__ import division
from parser_ import parser_
import networkx as nx
from math import *
import random
import copy
import re
import os

class Network:
	def __init__(self, max_dist, max_depth, option, filename):
		""" Creates the network G(V,E)  where the vertices (nodes) V correspond to junctions
		    and the edges E correspond to the pipes.

		    Sensor range - the extent of which a sensor can monitor for leakages
		                   i.e. a pipe is within a sensor's range if the sensor can detect
		                   a leakage event at that pipe

		    :param filename: A .net or .inp file containing the network topology
		    :param option: Option for distance-based modelling with the following values:
		                'distance': sensor range is base on the specified maximum
		                            distance
		                'depth': sensor range is based on the maximum depth
		                         (or number of hops)
		    :param max_dist: The maximum distance threshold for defining sensor
		    				 range.
		    :param max_depth: The maximum depth (number of hops) for defining
		    				  sensor range.
		 	"""
		
		self.filename = filename
		self.graph, self.position, self.lookup, self.nodes, self.edges, self.sfpd = self.initialize(filename)

		#Disance-based model
		self.max_dist = max_dist
		self.max_depth = max_depth
		self.range_list = self.readRangeList(os.path.basename(self.filename), option)

	def initialize(self, filename):
		""" Initializes the networkx graph and defines the lookup/ dictionaries for fast access.

			:param filename: A .net or .inp file containing the network topology
			:return graph: Networkx graph of the network
			:return position: A dictionary with
							  nodes as keys
							  coordinates as values
			:return lookup: A dictionary with
							edge-ids	  as keys
							(edge-tuples, edge_length) as values
			:return junctions: A list of all junction nodes (excluding reservoirs and tanks)
			:return pipe_ids: A list of all edge_ids of pipes (excluding pumps and valves)
		"""

		junctions, pipes, pumps, position, sfpd = parser_().parseInpFile(filename)
		edges = pipes + pumps
		pipe_ids = []
		graph = nx.MultiGraph()
		graph.add_nodes_from(junctions)

		# For fast access
		lookup = dict()
		for edge in edges:
			graph.add_edge(edge[1][0], edge[1][1], id=edge[0], length=edge[2])
			edge_tuple = (edge[1][0], edge[1][1])

			# Initilize lookup dictionary for fast access
			lookup[edge[0]] = (edge_tuple, edge[2])
			if edge in pipes:
				pipe_ids.append(edge[0])

		# Default value for locations with no flood level specified
		if sfpd:
			for node in junctions:
				if node not in sfpd:
					sfpd[node] = random.uniform(0, 0.1) #default

		return graph, position, lookup, junctions, pipe_ids, sfpd

	def readRangeList(self, base, option):
		"""
		Traverses the file 'archive' in folder models for the base (network filename)
		"""
		filename = 'models/archive'
		file = open(filename, 'r')

		range_list= None
		line = file.readline()
		while line:
			base_ = line.strip() #filename
			if base_ == base:
				range_list = dict()
				line = file.readline()
				while line:
					line = line.strip().split()
					range_list[line[0]] = []
					for edge in line[1:]:
						range_list[line[0]].append(edge)
					line = file.readline().strip()
					if line == '.':
						break
				break
			line = file.readline().strip()
		
		if range_list:
			return range_list
		else:
			print ("Constructing range list...")
			range_list = self.getRangeList(option)
			self.archiveRangeList(base, range_list)
			return range_list

	def archiveRangeList(self, base, range_list):
		"""
		Adds the range_list for network <base> to the file /models/archive
		The structure is as follows:
			<BASE>
			<JUNCTION_1 NAME> <LIST OF PIPE NAMES IN JUNCTION_1'S RANGE>
			...
			<JUNCTION_N NAME> <LIST OF PIPE NAMES IN JUNCTION_N'S RANGE>

		:param base: network filename (e.g. net1.inp)
		:param range_list: range_list to be archived
		"""

		filename = 'models/archive'
		with open(filename, 'a') as result:
			result.write(str(os.path.basename(base)) + "\n")
		for sensor in range_list:
			with open(filename, 'a') as result:
				result.write(sensor + " ")
			for edge in range_list[sensor]:
				with open(filename, 'a') as result:
					result.write(edge + " ")
			with open(filename, 'a') as result:
				result.write('\n')
		with open(filename, 'a') as result:
			result.write('.\n')

	def getTotalRange(self, sensors):
		""" Returns the total number of pipse that can be monitored by a set of sensors.
			:param sensors: A set of sensor locations (nodes).
			:return total_range: A set of pipes (edges) ith the sensor set's range.
		"""
		total_range = []
		for sensor in sensors:
			for target in self.range_list[sensor]:
				total_range.append(self.lookup[target][0])
		return total_range

	def getRangeList(self, option):
		""" 
		Distance-based model.
		:param option: Option for distance-based model with the following values:
		               'distance': sensor range is based on the specified maximum
		                           distance
		               'depth': sensor range is based on the maximum depth
		                         (number of hops)

		 :returns: A dictionary containing locations as keys and
		           the list of pipes within the sensor's range as values
		"""
		range_list = dict()

		def getObjectsWithinDistance(init, start, edge_ids):
			""" This algorithm recursively checks if neighboring nodes are
			    within the range of the initial node (init) and returns a
			    list of edge_ids that are within the range of init.

			    This distance-based model is based on the pipe lengths.
			"""

			neighbor_edges = list(self.graph.edges(start, data=True))
			neighbor_edges_ids = []

			def computeNNDistance(source, target):
				""" Computes Node-to-Node distance based on Dijkstra's shortest path algorithm"""

				shortest_path = nx.dijkstra_path(self.graph, source=source, target=target, weight='length')
				#shortest_dist =  nx.dijkstra_path_length(self.graph, source=source, target=target, weight='length')

				# Faster that using nx.dijkstra_path_length
				shortest_dist, length = 0, 0
				start = shortest_path[0]
				for node in shortest_path[1:]:
					length = self.graph.get_edge_data(start, node, default=0)[0]['length']
					shortest_dist += length
					start = node
				return shortest_dist, shortest_path

			def computeNEDistance(source, target_edge_id):
				""" Computes Node-to-Edge distance"""

				target_edge = self.lookup[target_edge_id][0]
				target1, target2 = target_edge[0], target_edge[1]
				dist1, path1 = computeNNDistance(source, target1)
				dist2, path2 = computeNNDistance(source, target2)
				edge_dist = self.lookup[target_edge_id][1]

				# Get distance from node to the center of the pipe
				# Both nodes target1 and target2 should be in the path; if not, append to path.
				total_dist = 0
				if target2 not in path1:
					dist1 = dist1 + (edge_dist/2) 
					path1.append(target2)
				elif target2 in path1:
					dist1 = dist1 - (edge_dist/2)

				if target1 not in path2:
					dist2 = dist2 + (edge_dist/2) 
					path2.append(target1)
				elif target1 in path2:
					dist2 = dist2 - (edge_dist/2)

				# Get the shorter distance
				total_dist = dist1 if  (dist1 < dist2) else dist2
				return total_dist

			# Get ids of the neighbors
			for neighbor_edge in neighbor_edges:
				neighbor_edges_ids.append(neighbor_edge[2]['id'])

			neighbor_nodes = []
			for edge_id in neighbor_edges_ids:
				distance = computeNEDistance(init, edge_id)
				if distance <= self.max_dist and edge_id not in edge_ids:
					edge_ids.append(edge_id)
					edge = self.lookup[edge_id][0]
					if start != edge[0]:
						neighbor_nodes.append(edge[0])
					else:
						neighbor_nodes.append(edge[1])

			for node in neighbor_nodes:
				getObjectsWithinDistance(init, node, edge_ids)
			return edge_ids

		def getObjectsInRange(init, start, edge_ids, depth):
			""" This algorithm recursively checks if neighboring nodes are
			    within the range of the initial node (init) and returns a
			    list of edge_ids that are within the range of init.

			    This distance-based model is based on the number of hops
			    (rather than the pipe lengths).
			"""

			if depth > self.max_depth:
				return

			# Get neighboring edges of start node
			# Note: Two different links that connect the same two nodes are counted twice
			neighbor_edges = list(self.graph.edges(start, data=True))
			neighbor_edges_ids = []

			# Get ids of the neighbors
			for neighbor_edge in neighbor_edges:
				neighbor_edges_ids.append(neighbor_edge[2]['id']) # one edge can have multiple ids

			# Append edge id to range
			neighbor_nodes = []
			for edge_id in neighbor_edges_ids:
				if edge_id not in edge_ids:
					edge_ids.append(edge_id)
					edge = self.lookup[edge_id][0] # Get corresponding edge of the id

					# Get all adjacent nodes from neighboring edges
					if start != edge[0]:
						neighbor_nodes.append(edge[0])
					else:
						neighbor_nodes.append(edge[1])

			# Recurse over neighboring nodes
			for neighbor_node in neighbor_nodes:
				getObjectsInRange(init, neighbor_node, edge_ids, depth+1)
			return edge_ids

		i = 0
		for node in self.graph.nodes():
			if option == 'depth':
				edge_ids = getObjectsInRange(node, node, [], 1)
			elif option == 'distance':
				edge_ids = getObjectsWithinDistance(node, node, [])
			i = i + 1
			if i % 10 == 0:
				print str(i) + "/" + str(len(self.nodes)) + " nodes complete..."

			# Makes sure all edges in the sensor range are pipes
			edge_ids_ = []
			for edge_id in edge_ids:
				if edge_id in self.edges: # all edges in self.edges are pipes
					edge_ids_.append(edge_id)
			range_list[node] = edge_ids_

		return range_list