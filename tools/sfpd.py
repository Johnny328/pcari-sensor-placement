import random
import math
import os

# Generates random sensor failure probabilities
def generateSFP(network, alpha):
	locations = network.nodes
	random.seed(64)
	sfpd = dict()

	while len(sfpd) < len(locations):
		rand_loc = random.choice(locations)
		rand = random.uniform(0, alpha)
		sfpd[rand_loc] = rand

		# Nodes that are within a 1km radius of the node 
		# are assigned the same sensor failure probability 
		# as that node
		for loc in locations:
			x1, y1 = network.position[rand_loc]
			x2, y2 = network.position[loc]
			dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			if dist < 1000:
				sfpd[loc] = rand
	return sfpd

def archiveSFPD(base, sfpd):
	filename = 'models/sfpd'

	with open(filename, 'a') as result:
		result.write(str(os.path.basename(base)) + "\n")
	for loc in sfpd:
		prob = sfpd[loc]
		with open(filename, 'a') as result:
			result.write(loc + " " + str(prob))
		with open(filename, 'a') as result:
			result.write('\n')
	with open(filename, 'a') as result:
		result.write('.\n')

def readSFPD(base):
		filename = 'models/sfpd'
		file = open(filename, 'r')

		sfpd= None
		line = file.readline()
		while line:
			base_ = line.strip() #filename
			if base_ == base:
				sfpd = dict()
				line = file.readline()
				while line:
					line = line.strip().split()
					sfpd[line[0]] = float(line[1])
					line = file.readline().strip()
					if line == '.':
						break
				break
			line = file.readline().strip()
		
		if sfpd:
			return sfpd