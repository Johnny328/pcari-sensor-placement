from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pylab import *
import networkx as nx
import ttk
import pylab
import matplotlib as mpl
from networkx import *
import copy
from colour import Color

def drawGraph(network, sensors, params, filename, subgraphs=None):
	sfpd = network.sfpd

	plt.clf()
	mpl.rcParams['text.usetex'] = True
	mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed
	totalrange = network.getTotalRange(sensors)

	pos = network.position
	fig, ax = plt.subplots(figsize=(10, 20))
	#nx.draw_networkx_edges(network.graph, network.position)
	#nx.draw_networkx_labels(network.graph,network.position, labels)

	newpos = dict()
	for sensor in pos:
		newpos[sensor] = tuple([x/1000 for x in pos[sensor]])

	# Set background colors according to sensor failure probability distribution
	for sensor in sfpd:
		if sensor in network.graph.nodes():
			if sfpd[sensor] < 0.1: color = '#ffff99' #yellow
			elif sfpd[sensor] < 0.2: color = '#ffc966' #orange
			else: color = '#fc8585' #red
			radius = 0.25
			if 'makati' in filename or 'sanjuan' in filename: radius = 0.05
			ax.add_artist(Circle(newpos[sensor], radius=radius, color=color, zorder=0))
			if sensor in sensors:
				width, height = 0.6, 0.6
				if 'makati' in filename or 'sanjuan' in filename: width, height = 0.1, 0.1
				ax.add_artist(Rectangle((newpos[sensor][0]- (width/2), newpos[sensor][1] - (height/2)),
							width=width, height=height, color = 'blue', linewidth=2, fill=False, zorder=5))
	

	nodes = [x for x in network.graph.nodes() if x not in sensors]
	nx.draw_networkx_nodes(network.graph, newpos, nodes, node_size=10, node_color='black')
	nx.draw_networkx_nodes(network.graph, newpos, nodelist=sensors, node_size=10, node_color='black')

	edges = [x for x in network.graph.edges() if x not in totalrange]
	nx.draw_networkx_edges(network.graph, newpos, edges, width=1, edge_color='black')
	nx.draw_networkx_edges(network.graph, newpos, totalrange, width=1, edge_color='black')

	x0, y0, dx, dy = ax.get_position().bounds
	maxd = max(dx, dy)
	width = 10*maxd /dx
	height = 10*maxd/dy
	
	plt.axis('scaled')
	fig.set_size_inches((width,height))

	plt.axis('off')
	n_text = 'No. of sensors:' + str(len(sensors)) + '\n'
	sub = "DP: " + str(round(params[0],4)) + "\nRisk: " + str(round(params[1],4)) + "\nRDP: " + str(round(params[2],4)) 
	figtext(.05, 0.8, n_text+sub, fontsize=30)
	plt.savefig(filename)
	#plt.show()
	plt.close()