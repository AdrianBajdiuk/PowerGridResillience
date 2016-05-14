import logging
import multiprocessing
from math import log
import numpy as np
from joblib import Parallel, delayed
from Helper import copyCase
from Helper import createRandomWalk
from Simulations import SimTask
import copy_reg
import types
import os
import time
import csv
from const import  constIn
import random


def _pickle_method(method):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


# This call to copy_reg.pickle allows you to pass methods as the first arg to
# mp.Pool methods. If you comment out this line, `pool.map(self.foo, ...)` results in
# PicklingError: Can't pickle <type 'instancemethod'>: attribute lookup
# __builtin__.instancemethod failed

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)



class SimProcessor(multiprocessing.Process):
    def __init__(self, inputQueue, outputQueue):
        multiprocessing.Process.__init__(self)
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue

    # take from queue
    # run SimTask
    # put to queue SimTask.getResult
    # def run(self):
    def run(self):
        while True:
            try:
                # grab task from input queue
                simTask = self.inputQueue.get()
                logging.log(logging.INFO,
                            "starting %(method)s method %(iter)d iteration" % {"method": simTask.method,
                                                                               "iter": simTask.iteration})
                simTask.run()
                result = simTask.getResult()
                logging.log(logging.INFO,
                            "finished with succes %(method)s method %(iter)d iteration with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                            {"method": simTask.method, "iter": simTask.iteration, "lcg": result[1], "pf": result[2]})
                self.outputQueue.put(("success",result))
            except Exception as x:
                logging.error(x.message)
                result = simTask.getResult()
                logging.log(logging.INFO,
                           "finished with error %(method)s method %(iter)d iteration with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                           {"method": simTask.method, "iter": simTask.iteration, "lcg": result[1], "pf": result[2]})
                self.outputQueue.put(("error",result))
            finally:
                self.inputQueue.task_done()


class MethodBase(multiprocessing.Process):
    # logging.basicConfig(filename="sim.log",level=logging.INFO,
    #                     format='%(asctime)s %(levelname)-8s %(message)s',
    #                     datefmt='%a, %d %b %Y %H:%M:%S')

    # destroyMethod = [VMaxK,VMaxPin,VRand,EMaxP,ERand]
    def __init__(self, processesCount,outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, vStep=None):
        multiprocessing.Process.__init__(self)
        self.outputDir = outputDir
        self.graphCopy = graphCopy
        self.caseCopy = caseCopy
        self.N = N
        self.methodName = methodName
        self.alpha = alpha
        self.destroyMethod = destroyMethod
        self.vStep = vStep
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.csvResult = []
        self.vizualizations = []
        self.serializationTime = time.strftime("%Y%m%d_%H-%M", time.localtime())
        self.simProcessors = []
        self.processesCount = processesCount

    def run(self):
        self.graphCopy, self.caseCopy = self.improveResiliency()
        vStep = self.vStep
        simTask = SimTask(self.methodName, 0, self.graphCopy.copy(), copyCase(self.caseCopy), v=vStep)
        self.tasks.put(simTask)
        # trigger of cascade, passed as reference to function
        for n in range(1, self.N):
            while True:
                # returns changed Pg,Pd, copies
                randomizedSimGraph,randomizedSimCase = self.randomizeGraphAndCase(self.graphCopy, self.caseCopy)
                randomizedSimTask = SimTask(self.methodName, n, randomizedSimGraph, randomizedSimCase)
                # checks overflows, after updating power flows
                if randomizedSimTask.isValid():
                    self.tasks.put(randomizedSimTask)
                    break
        # in this point tasks are generated
        # start processes, as much as cpu's
        num_cores = self.processesCount
        for p in range(num_cores):
            simP = SimProcessor(self.tasks, self.results)
            # self.simProcessors.append(simP)
            simP.daemon = True
            simP.start()

        # wait for all tasks to finish
        self.tasks.join()

        # serialize results
        self.extractResults(self.csvResult, self.vizualizations)
        self.serializeVisualizations(self.vizualizations[0])
        self.serializeCSV(self.csvResult)

        # after finishing, release simProcessors
        print "la"
        return

    def extractResults(self, csvResult, visualizations):

        # create output dir
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        for i in range(self.results.qsize()):
            res = self.results.get()
            if len(res[1][3]) > 0:
                visualizations += [res[1][3]]
            csvResult += [[res[0], res[1][0], res[1][1], res[1][2]]]

    # serialize results
    def serializeVisualizations(self, visualizations):
        visualizationsPath = os.path.join(self.outputDir, "visualizations")
        if not os.path.exists(visualizationsPath):
            os.makedirs(visualizationsPath)
        for v in visualizations:
            outputVisualizationpath = os.path.join(visualizationsPath,
                                                   str(v[0]) + "_" + self.serializationTime + ".GraphML")
            v[1].write_graphml(outputVisualizationpath)

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.methodName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status','n', 'LCCRatio', 'PfPdRatio', 'method', 'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status':i[0],'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3], 'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()
    # change Pin, and Pd to create another instance of case, but do not change topology
    # returns copies
    def randomizeGraphAndCase(self, graph, case):
        simGraph = graph.copy()
        simCase = copyCase(case)
        #update generated power
        for gen in simCase["gen"]:
            rand = random.uniform(-self.alpha,self.alpha)
            pg = gen[constIn["gen"]["Pg"]]
            pg += pg * rand
            genV = simGraph.vs.find(name="Bus_"+str(int(gen[constIn["gen"]["busIndex"]])))
            genV["Pg"] = pg
            gen[constIn["gen"]["Pg"]]= pg
        #update power demand
        for bus in simCase["bus"]:
            rand = random.uniform(-self.alpha,self.alpha)
            pd = bus[constIn["bus"]["Pd"]]
            pd += pd * rand
            busV = simGraph.vs.find(name="Bus_"+str(int(bus[constIn["bus"]["index"]])))
            busV["Pd"] = pd
            bus[constIn["bus"]["Pd"]] = pd
        return simGraph, simCase

    # checks if new instance is valid->no overflows!
    def isGraphAndCaseValid(self, graph, case):
        # cascadeStart param can be whatever because n=1
        simTask = SimTask(self.methodName, 0, graph, graph, graph, case, 0, n=1)
        return simTask.isValid()

    ## base function to override for all methods
    def improveResiliency(self):
        ##for base method return original graph
        return self.graphCopy, self.caseCopy


# class base for ESP from random_walkers
class ESPBase(MethodBase):
    def __init__(self,processesCount, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount,
                 improvement, vStep=None):
        MethodBase.__init__(self,processesCount, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, vStep)
        self.H = H
        self.M = M
        self.improvementCount = improvementCount
        self.improvement = improvement

    def etropyRow(self, a):
        f = np.vectorize(self.entropy)
        return -np.sum(f(a))

    def entropy(self, x):
        return x * log(x) if x != 0.0 else 0.0

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.methodName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status','n', 'LCCRatio', 'PfPdRatio', 'H', 'M', 'improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status':i[0],'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3], 'H': self.H, 'M': self.M,
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class ESPEdge(ESPBase):
    def __init__(self,processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount, improvement,
                 vStep=None):
        ESPBase.__init__(self,processesCount, outputDir, 'ESP edge', N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                         improvementCount, improvement, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(self.graphCopy.es.indices), len(vs)), dtype=float)
        num_cores = self.processesCount
        for vertex in vs:
            logging.log(logging.INFO, "starting walks for : " + str(vertex))
            walks = Parallel(n_jobs=num_cores)(
                delayed(createRandomWalk)(self.graphCopy, vertex.index, self.H) for i in range(0, self.M))
            for walk in walks:
                for i, e in enumerate([row[1] for row in walk]):
                    if (e is not None and i != (len(walk) - 1)):
                        espResult[e, vertex.index] += 1.0 / self.M
            logging.log(logging.INFO, "finished walks for : " + str(vertex))
        # print espResult
        results = np.apply_along_axis(self.etropyRow, 1, espResult)
        # selected for improvement edge indices
        selectedResults = (-results).argsort()[:self.improvementCount]
        logging.log(logging.INFO,"selected results: "+str(selectedResults))

        # improve c in selected edges

        for res in selectedResults:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += self.improvement

        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy


class ESPVertex(ESPBase):
    def __init__(self,processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount, improvement,
                 vStep=None):
        ESPBase.__init__(self,processesCount, outputDir, 'ESP vertex', N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                         improvementCount, improvement, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(vs), len(vs)), dtype=float)
        num_cores = self.processesCount
        for vertex in vs:
            logging.log(logging.INFO, "starting walks for : " + str(vertex))
            walks = Parallel(n_jobs=num_cores)(
                delayed(createRandomWalk)(self.graphCopy, vertex.index, self.H) for i in range(0, self.M))
            for walk in walks:
                for i, v in enumerate([row[0] for row in walk]):
                    if (v is not None and v != vertex.index):
                        espResult[v, vertex.index] += 1.0 / self.M
        # print espResult
        results = np.apply_along_axis(self.etropyRow, 1, espResult)
        # selected for improvement edge indices
        selectedResults = (-results).argsort()[:self.improvementCount]
        logging.log(logging.INFO, "selected results: " + str(selectedResults))
        for res in selectedResults:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += self.improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy


class RandomEdge(MethodBase):
    def __init__(self,processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvementCount, improvement,
                 vStep=None):
        MethodBase.__init__(self,processesCount, outputDir, 'Random edge', N, graphCopy, caseCopy, alpha, destroyMethod, vStep)
        self.improvementCount = improvementCount
        self.improvement = improvement

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        randomlyChosenEdges = np.random.choice(self.graphCopy.es.indices, self.improvementCount, replace=False)
        for res in randomlyChosenEdges:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += self.improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.methodName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio','improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status':i[0],'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class RandomVertex(MethodBase):
    def __init__(self,processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvementCount, improvement,
                 vStep=None):
        MethodBase.__init__(self, outputDir, 'Random vertex', N, graphCopy, caseCopy, alpha, destroyMethod, vStep)
        self.improvementCount = improvementCount
        self.improvement = improvement

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        randomlyChosenVertices = np.random.choice(self.graphCopy.vs.indices, self.improvementCount, replace=False)
        for res in randomlyChosenVertices:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += self.improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.methodName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio','improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status':i[0],'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()