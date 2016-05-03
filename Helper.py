##helpers for solving pf problem
from PowerGridResillience.const import constIn, constOut
from numpy import array, vstack, copy
from random import randint, choice
from itertools import islice, takewhile
from collections import Counter


# "bus":{"index":0,"type":1,"Pd":2,"Qd":3,"Vm":7,"Va":8,"baseKV":9,"area":6}
# "gen":{"busIndex":0,"Pq":1,"Qg":2,"Qmax":3,"Qmin":4,"Vg":5,"Pmax":8,"Pmin":9},
# "branch":{"fromBus":0,"toBus":1,"r":2,"x":3,"b":4}
def createCase(buses, generators, branches):
    ppc = {"version": 2}
    ppc["baseMVA"] = 100.0
    bus = array([])
    i = True
    for busIn in buses:
        if i:
            bus = array([busIn[0], busIn[1], busIn[2], busIn[3], 0, 0, busIn[7], busIn[4], busIn[5], busIn[6], 1, 1, 1])
            i = False
        else:
            bus = vstack(
                (bus, [busIn[0], busIn[1], busIn[2], busIn[3], 0, 0, busIn[7], busIn[4], busIn[5], busIn[6], 1, 1, 1]))
    ppc["bus"] = bus
    i = True
    gen = array([])
    for genIn in generators:
        if i:
            gen = array(
                [genIn[0], genIn[1], genIn[2], genIn[3], genIn[4], genIn[5], 100, 1, genIn[6], genIn[7], 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0])
            i = False
        else:
            gen = vstack((gen,
                          [genIn[0], genIn[1], genIn[2], genIn[3], genIn[4], genIn[5], 100, 1, genIn[6], genIn[7], 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0]))
    ppc["gen"] = gen
    i = True
    branch = array([])
    for branchIn in branches:
        if i:
            branch = array(
                [branchIn[0], branchIn[1], branchIn[2], branchIn[3], branchIn[4], 0, 0, 0, 0, 0, 1, -360, 360])
            i = False
        else:
            branch = vstack((branch,
                             [branchIn[0], branchIn[1], branchIn[2], branchIn[3], branchIn[4], 0, 0, 0, 0, 0, 1, -360,
                              360]))
    ppc["branch"] = branch
    return ppc


##deep copy of case for thread safety
def copyCase(case):
    ppc = {"version": 2}
    ppc["baseMVA"] = 100.0
    ppc["bus"] = copy(case["bus"])
    ppc["gen"] = copy(case["gen"])
    ppc["branch"] = copy(case["branch"])
    return ppc;


def random_walk_iter(g, start=None):
    previousVertices = []
    currentVerticeIndex = randint(0, g.vcount() - 1) if start is None else start
    while True:
        possibleEdgesIndices = (
            Counter(g.es.select(_source=currentVerticeIndex).indices + g.es.select(
                _target=currentVerticeIndex).indices) - Counter(
                g.es.select(_source_in=previousVertices).indices + g.es.select(
                    _target_in=previousVertices).indices)).keys()
        if (currentVerticeIndex is not None and len(possibleEdgesIndices) == 0):
            yield currentVerticeIndex, None
            previousVertices = previousVertices + [currentVerticeIndex]
            currentVerticeIndex = None
        elif (currentVerticeIndex is None and len(possibleEdgesIndices) == 0):
            yield None, None
        else:
            currentEdgeIndex = choice(possibleEdgesIndices)
            yield [currentVerticeIndex, currentEdgeIndex]
            previousVertices = previousVertices + [currentVerticeIndex]
            edge = g.es.find(currentEdgeIndex)
            currentVerticeIndex = edge.target if edge.source == currentVerticeIndex else edge.source


def createRandomWalk(g, start, H):
    return list(islice(random_walk_iter(g, start), H))

def increaseEdgeC(edge,graph,increaseValue):
    graph.es.find(int(edge))["c"] += increaseValue
def increaseVertexC(vertex,graph,increaseValue):
    graph.vs.find(int(vertex))["c"] += increaseValue