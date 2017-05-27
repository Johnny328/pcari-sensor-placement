METER_CONV = 0.3048
SCALE = 1.07

class parser_:
    """
    Parses through .INP EPANET files
    :param filename: A .inp file (see /models folder for examples)

    :return junctions: A list of nodes (not including tanks and reservoirs)
    :return pipes: A list of edges (not including pumps and valves)
    :return pumps: A list of pumps and valves (of zero length)
    :return pos: A dictionary of positions of each node in junctions
    """
    def parseInpFile(self, filename):
        ign = ';'
        titles = ['[PIPES]', '[VALVES]', '[PUMPS]', '[JUNCTIONS]', '[RESERVOIRS]', '[TANKS]', '[COORDINATES]', '[TAGS]', '[DEMANDS]']
        end = '[END]'
        comps = {}  # components

        inputfile = open(filename)
        line = inputfile.readline().strip()

        while line.strip() != end:
            if line.strip() in titles:
                newset = []
                nextline = inputfile.readline()
                while nextline.strip():
                    if (ign in nextline.split()[0]):
                        nextline = inputfile.readline()
                    newset.append(nextline.split())
                    nextline = inputfile.readline()
                    if nextline.strip() == end:
                        break
                    if nextline.strip() in titles:
                        break
                comps[line.strip()] = newset
                line = nextline
            else:
                line = inputfile.readline()

        junctions, pipes, pumps, pos, sfpd = [], [], [], {}, None
        for title in titles:
            if title in comps:
                if title == '[JUNCTIONS]': #or title == '[RESERVOIRS]' or  title == '[TANKS]':
                    for node in comps[title]:
                        junctions.append(node[0])
                elif title == '[PIPES]':
                    for edge in comps[title]:
                        edge_pair = (edge[1], edge[2])
                        if 'makati' in filename:
                            distance =  float(edge[3])*(SCALE)
                        else:
                            distance =  float(edge[3])*(METER_CONV)
                        pipes.append([edge[0], edge_pair, distance])
                elif title == '[PUMPS]' or title == '[VALVES]':
                    for edge in comps[title]:
                        if edge:
                            edge_pair = (edge[1], edge[2])
                            distance =  float(0)
                            pumps.append([edge[0], edge_pair, distance])
                elif title == '[COORDINATES]':
                    for node in comps[title]:
                        pos[node[0]] = (float(node[1]),float(node[2]))
                elif title == '[TAGS]':
                    sfpd = {}
                    for line in comps[title]:
                        code = line[2]
                        if code == 'S':
                            sfpd[line[1]] = 0.3
                        elif code == 'F':
                            sfpd[line[1]] = 0.2

        inputfile.close()
        return junctions, pipes, pumps, pos, sfpd

    def parserSetSensors(self, nodes):
        sensor_str = 'SENSOR'
        sensors = []
        for node in nodes:
            if sensor_str in node:
                sensors.append(node)
        return sensors

    def readResults(self, filename):
        inputfile = open(filename)
        k, dp = [], []

        line = inputfile.readline()
        while line.strip():
            k.append(line.split()[0])
            dp.append(line.split()[1])
            line = inputfile.readline()

        inputfile.close()
        return k, dp
