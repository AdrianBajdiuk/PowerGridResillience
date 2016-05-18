import multiprocessing
import threading
import logging
from runPfOptions import ppoption
from pypower.api import runpf
from const import constOut
import random
import math
import sys
import numpy as np
from GraphCaseUpdates import deleteVertice,deleteEdge

class SimTask:
    def __init__(self, methodName, iteration, graphCopy,caseCopy, n=None, v=None):
        self.method = methodName
        # needed for simulation
        self.graph = graphCopy
        # needed for calculating results
        self.referenceGraph = self.graph.copy()
        # needed for calculating visualizations
        self.graphVisual = self.graph.copy()
        # needed for calculating prob.
        self.previousGraph = self.graph.copy()
        self.iteration = iteration
        self.case = caseCopy
        self.n = n
        self.v = v
        self.cascadeTrigger = None
        self.visualize = not (self.v is None)
        self.visualizations = [[-1,self.graphVisual.copy()]] if self.visualize else []
    #can be checked only once at beginning
    def isValid(self):
        logging.log(logging.INFO,
                    "starting %(method)s method %(iter)d case validity check" % {"method": self.method,
                                                                                 "iter": self.iteration})
        self.updateGraphFlow(self.referenceGraph,self.case)
        # cascade start can be whatever
        malfunctions = self.findMalfunctions(self.referenceGraph,self.previousGraph,0)
        result = len(malfunctions)>0
        if result:
            logging.log(logging.INFO,
                    " %(method)s method %(iter)d case validity check failed, generate again" % {"method": self.method,
                                                                                 "iter": self.iteration})
        return not result

    def runSimulation(self):

        logging.log(logging.INFO,
                    "starting %(method)s method simulation in %(iter)d iteration" % {"method": self.method, "iter": self.iteration})
        try:
            self.updateGraphFlow(self.graphVisual,self.case)
            self.updateGraphFlow(self.graph, self.case)
            # self.getResult()
            # destroy the graph
            self.graph, self.case, self.graphVisual, self.cascadeTrigger = self.destroyGraph(self.graph, self.case,
                                                                                             self.graphVisual)
        except SimException as e:
            logging.error(e)
            raise e
        if not (self.n is None):
            # for specified count
            logging.log(logging.INFO, "for specified  %(count)d iterations" % {"count": self.n})
            for iter in range(0,self.n):
                self.updateGraphFlow(self.graph,self.case)
                malfunctions = self.findMalfunctions(self.graph,self.previousGraph)
                if len(malfunctions) > 0:
                    self.simulate(iter, malfunctions)
                else:
                    logging.log(logging.INFO, "found  0 overflows for %(method)s method in %(iter)d iteration."
                                              " Terminating" % {"method": self.method, "iter": self.iteration})
                    return
        else:
            # until no malfunctions stops
            counter = 0;
            while True:
                try:
                    self.updateGraphFlow(self.graph,self.case)
                    malfunctions = self.findMalfunctions(self.graph, self.previousGraph, self.cascadeTrigger)
                    if len(malfunctions) > 0:
                        self.simulate(counter, malfunctions)
                        if self.visualize and (counter == 0 or (counter % self.v) == 0):
                            self.visualizations += [[counter, self.graphVisual.copy()]]
                        counter += 1
                    else:
                        logging.log(logging.INFO, "found  0 overflows for %(method)s method in %(iter)d iteration, try %(try)d."
                                              " Terminating" % {"method": self.method, "iter": self.iteration,
                                                                "try": counter})
                        return
                except SimException as e:
                    logging.error(e)
                    raise e

    def simulate(self, counter, malfunctions):
        logging.log(logging.INFO, "found  %(over)d overflows for %(method)s method"
                                  " in %(iter)d iteration %(try)d try"
                    % {"over": len(malfunctions), "method": self.method, "iter": self.iteration,
                       "try": counter})
        # calculate Probablity based on shortest path lengths
        # calculate sum of all path lenghts:
        sum = 0.0
        for mal in malfunctions:
            try:
                mal["p"] = 1.0 / float(mal["pathL"])
            except ZeroDivisionError:
                mal["p"] = 0.0
            sum += mal["p"]
        selected = None
        if sum != 0.0:
            # calculate normalized p
            for mal in malfunctions:
                mal["p"] = mal["p"] / sum
            # assign lower and uper bounds of p for roulette
            malfunctions[0]["lower"] = 0.0
            malfunctions[0]["upper"] = 0.0 + malfunctions[0]["p"]
            for i, item in enumerate(malfunctions):
                if i > 0:
                    item["lower"] = malfunctions[i - 1]["upper"]
                    item["upper"] = item["lower"] + item["p"]
            malfunctions[-1]["upper"] = 1.0
            p = random.random()
            selected = next((x for x in malfunctions if ((x["lower"] <= p) and (x["upper"] > p))), None)
        else:
            selected = malfunctions[random.randint(0, len(malfunctions) - 1)]
        if selected is not None:
            # save previous graph for paths:
            self.previousGraph = self.graph.copy()
            logging.log(logging.INFO, "removing  %(name)s of type %(type)s from graph in  %(method)s method"
                                      " %(iter)d iteration %(try)d try"
                        % {"name": selected["name"], "type": selected["type"], "method": self.method,
                           "iter": self.iteration, "try": counter})

            # choosen to delete vertice
            if selected["type"] is "v":
                self.cascadeTrigger = selected["index"]
                vName = self.graph.vs.find(selected["index"])["name"]
                self.graph, self.case,self.graphVisual = deleteVertice(vName, self.graph, self.case, self.graphVisual)
            elif selected["type"] is "e":
                # 1st delete edge, check ,after delete all vertices with degree=0
                self.cascadeTrigger = self.graph.es.find(selected["index"]).target
                self.graph, self.case,self.graphVisual = deleteEdge(selected["index"], self.graph, self.case, self.graphVisual)

    # updates the flows in a graph
    def updateGraphFlow(self,graph,case):
        solverResult = None
        try:
            solverResult = runpf(case, ppopt=ppoption())
            # get from the results branches with counters if there is more branches between two buses
            counters = []
            i = 0
            for branch in solverResult[0]["branch"]:
                fromBusTemp = int(branch[constOut["branch"]["fromBus"]])
                toBusTemp = int(branch[constOut["branch"]["toBus"]])
                pIn = branch[constOut["branch"]["Pin"]]
                first_or_default = next(
                    (x for x in counters if x["fromBus"] == fromBusTemp and x["toBus"] == toBusTemp),
                    None)
                if first_or_default is None:
                    counters.append({"fromBus": fromBusTemp, "toBus": toBusTemp, "counter": 0, "Pin": pIn, "index": i})
                else:
                    counters.append(
                        {"fromBus": fromBusTemp, "toBus": toBusTemp, "counter": first_or_default["counter"] + 1,
                         "Pin": pIn,
                         "index": i})
                i = i + 1

            # for all vertices assign 0.0 for Pin and 0.0 for Pg
            for v in graph.vs:
                v["Pin"] = 0.0
                v["Pout"] = 0.0
                v["Pg"] = 0.0
            for gen in solverResult[0]["gen"]:
                busName = "Bus_" + str(int(gen[0]))
                graph.vs.find(busName)["Pg"] = gen[1]

            for c in counters:
                f = graph.vs.find("Bus_" + str(c["fromBus"]))
                t = graph.vs.find("Bus_" + str(c["toBus"]))
                graph.es.select(_source=f.index, _target=t.index)[c["counter"]]["Pin"] = c["Pin"]
                if (c["Pin"] > 0):
                    t["Pin"] += c["Pin"]
                    f["Pout"] += c["Pin"]
                else:
                    f["Pin"] += math.fabs(c["Pin"])
                    t["Pout"] += math.fabs(c["Pin"])
        except ValueError as ve:
            logging.error(ve)
            raise SimException(ve.message)

    # returns vertices and edges indexes over c, with p to be next
    def findMalfunctions(self, graph, previousGraph, cascadeTriggerInPreviousStep):
        overFlows = []
        graphUndirected = previousGraph.as_undirected()
        for edge in graph.es:
            if math.fabs(edge["Pin"]) > edge["c"] and not edge["c"] == 0.0 and edge["deletable"]:  # never,ever,destroy slack bus connections!
                prevGraphCurrentEdgeTargetIndex = previousGraph.vs.find(
                    name=graph.vs.find(edge.target)["name"]).index
                pathL = len(graphUndirected.get_shortest_paths(cascadeTriggerInPreviousStep,
                                                               prevGraphCurrentEdgeTargetIndex)[0])
                if pathL == 0:
                    pathL = sys.float_info.max
                edgeFrom = graph.vs.find(edge.source)["name"]
                edgeTo = graph.vs.find(edge.target)["name"]
                overFlows.append({"index": edge.index,
                                  "pathL": pathL,
                                  "type": "e",
                                  "name":edgeFrom+"-> "+edgeTo})
        for vertice in graph.vs:
            if (vertice["Pin"] > vertice["c"] and vertice["deletable"]):  # never,ever,destroy slack bus !
                prevGraphCurrentVerticeIndex = previousGraph.vs.find(name=vertice["name"]).index
                pathL = len(graphUndirected.get_shortest_paths(cascadeTriggerInPreviousStep,
                                                           prevGraphCurrentVerticeIndex)[0])
                if pathL == 0:
                    pathL = sys.float_info.max
                overFlows.append({"index": vertice.index,
                                  "pathL": pathL,
                                  "type": "v",
                                  "name": vertice["name"]})

        return overFlows

    def getResult(self):
        largestConnectedComponentsRatio = float(len(self.graph.components(mode='WEAK').giant().vs)) / float(
            len(self.referenceGraph.components(mode='WEAK').giant().vs))
        actualFlow = sum([self.actualPflowForVertex(v) for v in self.graph.vs])
        actualFlowPdRatio = actualFlow / sum(self.referenceGraph.vs["Pd"])
        return self.iteration,largestConnectedComponentsRatio,actualFlowPdRatio, self.visualizations

    # destroy method default VmaxK
    def destroyGraph(self, graph, case , graphVisual,destroyMethod=None):
        # copy self.graphCopy
        # destroy, based on max k
        graphCascadeTrigger = graph.vs.select(_degree=graph.maxdegree())[0].index
        graph, case, graphVisual = deleteVertice(graphCascadeTrigger, graph, case, graphVisual)
        return graph, case, graphVisual, graphCascadeTrigger

    def actualPflowForVertex(self,v):

        res = v["Pin"]+ v["Pg"]-v["Pout"]
        if res>0.0 :
            return res
        else:
            return 0.0
class SimException(Exception):
    pass