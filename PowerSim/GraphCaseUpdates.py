import numpy as np
import random
from const import constIn
# deletes given verticeIndex from graph and case, and update graphBase
# returns graph,case
def deleteVertice(verticeName, graph, case, graphBase=None):
    v = graph.vs.find(name=verticeName)
    vName = v["name"]
    caseBusIndex = int(vName.split("_")[-1])
    # remove branches from case
    case["branch"] = case["branch"][case["branch"][:, 0] != caseBusIndex]
    case["branch"] = case["branch"][case["branch"][:, 1] != caseBusIndex]
    # remove bus from case
    case["bus"] = case["bus"][case["bus"][:, 0] != caseBusIndex]
    # remove gen from case
    case["gen"] = case["gen"][case["gen"][:, 0] != caseBusIndex]
    # remove vertice and edges from graph
    graph.delete_vertices(v.index)
    if graphBase is not None:
        vGraphBase = graphBase.vs.find(name=vName)
        vGraphBase["isDeleted"] = True
        edgesIndices = graphBase.es.select(_source=vGraphBase.index).indices + graphBase.es.select(
            _target=vGraphBase.index).indices
        [setEdgeDeleted(graphBase,edge) for edge in edgesIndices]
    zeroDegreeNames = graph.vs.select(_degree=0)["name"]
    if len(zeroDegreeNames)>0:
        return deleteVertice(zeroDegreeNames[0],graph,case,graphBase)
    return graph, case, graphBase


def setEdgeDeleted(g, edgeIndex):
    g.es.find(edgeIndex)["isDeleted"] = True


# deletes given edge grom graph and case, and update graphBase
# returns graph,case
def deleteEdge(edgeIndex, graph, case, graphBase=None):
    fVertIndex = graph.es.find(edgeIndex).source
    tVertIndex = graph.es.find(edgeIndex).target
    fVertName = graph.vs.find(fVertIndex)["name"]
    tVertName = graph.vs.find(tVertIndex)["name"]
    fCaseIndex = int(fVertName.split("_")[-1])
    tCaseIndex = int(tVertName.split("_")[-1])
    # find all possible edges between those vertices
    # find position of edge under deletion
    # delete from branch at the same ranked position
    edgeGraphPosition = graph.es.select(_source=fVertIndex, _target=tVertIndex).indices \
        .index(edgeIndex)
    # np.where((case["branch"][:,0] == 1) & ((case["branch"][:,1]==2)))[0]
    edgeCasePosition = np.where((case["branch"][:, 0] == fCaseIndex) &
                                ((case["branch"][:, 1] == tCaseIndex)))[0][edgeGraphPosition]
    case["branch"] = np.delete(case["branch"], edgeCasePosition, 0)
    graph.delete_edges(edgeIndex)
    if graphBase is not None:
        vGraphbaseF = graphBase.vs.find(name=fVertName)
        vGraphBaseT = graphBase.vs.find(name=tVertName)
        edge = graphBase.es.select(_source =vGraphbaseF.index, _target=vGraphBaseT.index)[edgeGraphPosition]
        setEdgeDeleted(graphBase,edge.index)
    zeroDegreeNames = graph.vs.select(_degree=0)["name"]
    if len(zeroDegreeNames)>0:
        return deleteVertice(zeroDegreeNames[0],graph,case,graphBase)

    return graph, case, graphBase

def randomizePG(gen, graph, alpha):
    rand = random.uniform(-alpha, alpha)
    pg = gen[constIn["gen"]["Pg"]]
    pg += rand*pg
    genV = graph.vs.find("Bus_" + str(int(gen[constIn["gen"]["busIndex"]])))
    genV["Pg"] = pg
    gen[constIn["gen"]["Pg"]] = pg
    return pg


def randomizePD(self, bus, graph, alpha):
    rand = random.uniform(alpha, alpha)
    pd = bus[constIn["bus"]["Pd"]]
    pd += rand*pd
    busV = graph.vs.find("Bus_" + str(int(bus[constIn["bus"]["index"]])))
    busV["Pd"] = pd
