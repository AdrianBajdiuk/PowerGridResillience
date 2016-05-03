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
            # grab task from input queue
            simTask = self.inputQueue.get()
            logging.log(logging.INFO,
                        "starting %(method)s method %(iter)d iteration" % {"method": simTask.method,
                                                                           "iter": simTask.iteration})
            simTask.run()
            result = simTask.getResult()
            logging.log(logging.INFO,
                        "finished %(method)s method %(iter)d iteration with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                        {"method": simTask.method, "iter": simTask.iteration, "lcg": result[1], "pf": result[2]})
            self.outputQueue.put(result)
            self.inputQueue.task_done()


class MethodBase(multiprocessing.Process):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # destroyMethod = [VMaxK,VMaxPin,VRand,EMaxP,ERand]
    def __init__(self, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, vStep=None):
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

    def run(self):
        self.graphCopy, self.caseCopy = self.improveResiliency()
        vStep = self.vStep
        baseSimGraph = self.graphCopy.copy()
        baseSimCase = self.caseCopy.copy()
        simTask = SimTask(self.methodName, 0, baseSimGraph, baseSimCase, v=vStep)
        self.tasks.put(simTask)
        # trigger of cascade, passed as reference to function
        for n in range(1, self.N):
            while True:
                # returns changed Pg,Pd
                simGraph, simTask = self.randomizeGraphAndCase(self.graphCopy, self.caseCopy)
                randomizedSimTask = SimTask(self.methodName, n, simGraph, simTask)
                # checks overflows, after updating power flows
                if randomizedSimTask.isValid():
                    self.tasks.put(randomizedSimTask)
                    break
        # in this point tasks are generated
        # start processes, as much as cpu's
        num_cores = multiprocessing.cpu_count()
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
            if len(res[3]) > 0:
                visualizations += [res[3]]
            csvResult += [[res[0], res[1], res[2]]]

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
            fieldNames = ['n', 'LCCRatio', 'PfPdRatio', 'method', 'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'n': i[0], 'LCCRatio': i[1], 'PfPdRatio': i[2], 'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]

    # change Pin, and Pd to create another instance of case, but do not change topology
    # returns copies
    def randomizeGraphAndCase(self, graph, case):
        return graph.copy(), copyCase(case)

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
    def __init__(self, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount,
                 improvement, vStep=None):
        MethodBase.__init__(self, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, vStep)
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
            fieldNames = ['n', 'LCCRatio', 'PfPdRatio', 'H', 'M', 'improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'n': i[0], 'LCCRatio': i[1], 'PfPdRatio': i[2], 'H': self.H, 'M': self.M,
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]


class ESPEdge(ESPBase):
    def __init__(self, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount, improvement,
                 vStep=None):
        ESPBase.__init__(self, outputDir, 'ESP edge', N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                         improvementCount, improvement, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(self.graphCopy.es.indices), len(vs)), dtype=float)
        num_cores = multiprocessing.cpu_count()
        for vertex in vs:
            walks = Parallel(n_jobs=num_cores)(
                delayed(createRandomWalk)(self.graphCopy, vertex.index, self.H) for i in range(0, self.M))
            for walk in walks:
                for i, e in enumerate([row[1] for row in walk]):
                    if (e is not None and i != (len(walk) - 1)):
                        espResult[e, vertex.index] += 1.0 / self.M
        # print espResult
        results = np.apply_along_axis(self.etropyRow, 1, espResult)
        # selected for improvement edge indices
        selectedResults = (-results).argsort()[:self.improvementCount]
        # improve c in selected edges

        for res in selectedResults:
            edge = self.graphCopy.es.find(int(res))
            edge["c"] += self.improvement

        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy


class ESPVertex(ESPBase):
    def __init__(self, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M, improvementCount, improvement,
                 vStep=None):
        ESPBase.__init__(self, outputDir, 'ESP vertex', N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                         improvementCount, improvement, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(vs), len(vs)), dtype=float)
        num_cores = multiprocessing.cpu_count()
        for vertex in vs:
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
        for res in selectedResults:
            vertex = self.graphCopy.vs.find(int(res))
            vertex["c"] += self.improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy


class RandomEdge(MethodBase):
    def __init__(self, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvementCount, improvement,
                 vStep=None):
        MethodBase.__init__(self, outputDir, 'Random edge', N, graphCopy, caseCopy, alpha, destroyMethod, vStep)
        self.improvementCount = improvementCount
        self.improvement = improvement

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        randomlyChosenEdges = np.random.choice(self.graphCopy.es.indices, self.improvementCount, replace=False)
        for res in randomlyChosenEdges:
            edge = self.graphCopy.es.find(int(res))
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
            fieldNames = ['n', 'LCCRatio', 'PfPdRatio', 'H', 'M', 'improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'n': i[0], 'LCCRatio': i[1], 'PfPdRatio': i[2],
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]


class RandomVertex(MethodBase):
    def __init__(self, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvementCount, improvement,
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
            fieldNames = ['n', 'LCCRatio', 'PfPdRatio', 'H', 'M', 'improvement', 'improvementCount', 'method',
                          'destroyMethod', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'n': i[0], 'LCCRatio': i[1], 'PfPdRatio': i[2],
                              'improvement': self.improvement, 'improvementCount': self.improvementCount,
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'alpha': self.alpha}) for i in csvResult]