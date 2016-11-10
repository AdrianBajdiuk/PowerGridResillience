import logging
import multiprocessing
from math import log, floor,fabs
import numpy as np
from joblib import Parallel, delayed
from Helper import copyCase, createRandomWalk, configureBasicLogger,harmonicClosenessForV,shortestPathsParticipationforE
from Simulations import SimTask
import copy_reg
import types
import os
import time
import csv
from const import constIn
import random
import quehandler
from time import strftime, gmtime



# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them,
def listener_process(queue, logDir, logPath):
    # configurer()
    root = logging.getLogger()
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        # flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        # fd = os.open(logPath, flags)
        # os.close(fd)
    fileHandler = logging.FileHandler(logPath)
    # h = logging.handlers.RotatingFileHandler('mptest.log', 'a', 300, 10)
    f = logging.Formatter('%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] %(message)s',
                          datefmt='%m-%d %H:%M:%S')
    fileHandler.setFormatter(f)
    root.addHandler(fileHandler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # tell the handler to use this format
    console.setFormatter(f)
    # add the handler to the root logger
    root.addHandler(console)

    while True:
        try:
            record = queue.get()
            if record is not None:
                logger = logging.getLogger(record.name)
                logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            # print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# class SimProcessor(multiprocessing.Process):
#     def __init__(self, inputQueue, outputQueue,logQueue):
#         multiprocessing.Process.__init__(self)
#         self.inputQueue = inputQueue
#         self.outputQueue = outputQueue
#         h = quehandler.QueueHandler(logQueue)  # Just the one handler needed
#         root = logging.getLogger()
#         root.addHandler(h)
#         # send all messages, for demo; no other level or filter logic applied.
#         root.setLevel(logging.DEBUG)


def simProcess(inputQueue, outputQueue, logQueue):
    # configure handlers
    configureProcessLogger(logQueue)

    # take from queue
    # run SimTask
    # put to queue SimTask.getResult
    # def run():
    while True:
        name = multiprocessing.current_process().name
        isAlive = multiprocessing.current_process().is_alive()
        logger = logging.getLogger()
        logger.log(logging.INFO,
                   "current simTasks queue size is %(size)d, process name %(name)s" % {
                       "size": inputQueue.qsize(),
                       "name": name})
        logger.log(logging.INFO, "am alive? " + str(isAlive))
        # grab task from input queue
        simTask = None
        try:
            simTask = inputQueue.get()
            if simTask is not None:
                logger.log(logging.INFO,
                           "starting %(method)s method %(iter)d iteration" % {"method": simTask.method,
                                                                              "iter": simTask.iteration})
                simTask.runSimulation()
                result = simTask.getResult()
                logger.log(logging.INFO,
                           "finished with succes %(method)s method %(iter)d iteration with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                           {"method": simTask.method, "iter": simTask.iteration, "lcg": result[1],
                            "pf": result[2]})
                outputQueue.put(("success", result))
        except Exception as x:
            logger.error(x)
            # result = simTask.getResult()
            result = {-1, -1, -1, -1}
            if simTask is not None:
                logger.log(logging.INFO,
                           "finished with error %(method)s method %(iter)d iteration with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                           {"method": simTask.method, "iter": simTask.iteration, "lcg": result[1],
                            "pf": result[2]})
            else:
                logger.log(logging.INFO,
                           "finished with error ")
            outputQueue.put(("error", result))
        finally:
            if simTask is not None:
                inputQueue.task_done()


def configureProcessLogger(logQueue):
    h = quehandler.QueueHandler(logQueue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


# change Pin, and Pd to create another instance of case, but do not change topology
# returns copies
# randomize case and graph:
def randomizeGraphAndCase(i, methodName, graphCopy, caseCopy, alpha,vMaxK):
    logging.log(logging.INFO,
                "starting %(method)s method %(iter)d case validity check" % {"method": methodName,
                                                                             "iter": i})
    while True:
        simGraph = graphCopy.copy()
        simCase = copyCase(caseCopy)
        # update generated power
        for gen in simCase["gen"]:
            rand = random.uniform(-alpha, alpha)
            pg = gen[constIn["gen"]["Pg"]]
            pg += pg * rand
            genV = simGraph.vs.find(name="Bus_" + str(int(gen[constIn["gen"]["busIndex"]])))
            genV["Pg"] = pg
            gen[constIn["gen"]["Pg"]] = pg
        # update power demand
        for bus in simCase["bus"]:
            rand = random.uniform(-alpha, alpha)
            pd = bus[constIn["bus"]["Pd"]]
            pd += pd * rand
            busV = simGraph.vs.find(name="Bus_" + str(int(bus[constIn["bus"]["index"]])))
            busV["Pd"] = pd
            bus[constIn["bus"]["Pd"]] = pd

        randomizedSimTask = SimTask(methodName, i, simGraph, simCase,vMaxK=vMaxK)
        # checks overflows, after updating power flows
        if randomizedSimTask.isValid():
            return randomizedSimTask
        logging.log(logging.INFO,
                    " %(method)s method %(iter)d case validity check failed, generate again" % {"method": methodName,
                                                                                                "iter": i})


class MethodBase(multiprocessing.Process):
    def __init__(self, dataName, simName, processesCount, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod,
                 vMaxK=1,vStep=None):
        multiprocessing.Process.__init__(self)
        self.dataName = dataName
        self.simName = simName
        dataName = dataName if not os.path.isabs(dataName) else os.path.basename(dataName)
        self.outputDir = os.path.join(outputDir,dataName)
        self.graphCopy = graphCopy
        self.caseCopy = caseCopy
        self.N = N
        self.methodName = methodName
        self.alpha = alpha
        self.destroyMethod = destroyMethod
        self.vStep = vStep
        self.vMaxK = vMaxK
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.csvResult = []
        self.vizualizations = []
        self.serializationTime = time.strftime("%Y%m%d_%H-%M", time.localtime())
        self.simProcessors = []
        self.processesCount = processesCount


    def run(self):
        # logging queue
        configureBasicLogger(self.outputDir, "sim-" + self.dataName + "-" + self.serializationTime + ".log")
        logging.log(logging.INFO, "starting %s task with data %s" % (self.simName, self.dataName))
        logQueue = multiprocessing.Queue(-1)
        simLogPath = os.path.join(self.outputDir, self.simName + "_" + strftime("%H-%M", gmtime()) + ".log")
        logListener = multiprocessing.Process(target=listener_process, args=(logQueue, self.outputDir, simLogPath))
        logListener.daemon = True
        logListener.start()

        self.graphCopy, self.caseCopy = self.improveResiliency()
        vStep = self.vStep
        simTask = SimTask(self.methodName, 0, self.graphCopy.copy(), copyCase(self.caseCopy),vMaxK=self.vMaxK, v=vStep)
        self.tasks.put(simTask)
        # trigger of cascade, passed as reference to function

        logging.log(logging.INFO,
                    "starting %(method)s method generation of %(iter)d probes" % {"method": self.methodName,
                                                                                  "iter": self.N - 1})
        randomizedSimTasks = Parallel(n_jobs=self.processesCount, verbose=50)(
            delayed(randomizeGraphAndCase)(i, self.methodName, self.graphCopy.copy(), copyCase(self.caseCopy),
                                           self.alpha,self.vMaxK)for i in range(1, self.N))

        logging.log(logging.INFO,
                    "finished %(method)s method generation of %(iter)d probes" % {"method": self.methodName,
                                                                                  "iter": self.N - 1})

        # for n in range(1, self.N):
        #     while True:
        #         # returns changed Pg,Pd, copies
        #         randomizedSimGraph, randomizedSimCase = self.randomizeGraphAndCase(self.graphCopy, self.caseCopy)
        #         randomizedSimTask = SimTask(self.methodName, n, randomizedSimGraph, randomizedSimCase)
        #         # checks overflows, after updating power flows
        #         if randomizedSimTask.isValid():
        #             self.tasks.put(randomizedSimTask)
        #             break

        # generate tasks paralelly,and append to simTasks
        for task in randomizedSimTasks:
            self.tasks.put(task)

        # in this point tasks are generated
        # start processes, as much as configured
        for p in range(self.processesCount):
            simP = multiprocessing.Process(target=simProcess, args=(self.tasks, self.results, logQueue))
            # self.simProcessors.append(simP)
            simP.daemon = True
            simP.start()
            # time.sleep(2)

        # wait for all tasks to finish
        self.tasks.join()

        # serialize results
        self.extractResults(self.csvResult, self.vizualizations)
        self.serializeVisualizations(self.vizualizations[0])
        self.serializeCSV(self.csvResult)

        # after finishing, release simProcessors, auto because daemons
        print "la"
        return

    def extractResults(self, csvResult, visualizations):

        # create output dir
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        for i in range(self.results.qsize()):
            res = self.results.get()
            if len(res[1][4]) > 0:
                visualizations += [res[1][4]]
            csvResult += [[res[0], res[1][0], res[1][1], res[1][2], res[1][3]]]

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
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'method', 'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3], 'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

    # checks if new instance is valid->no overflows!
    def isGraphAndCaseValid(self, graph, case):
        # cascadeStart param can be whatever because n=1
        simTask = SimTask(self.methodName, 0, graph, graph, graph, case, 0, n=1)
        return simTask.isValid()

    ## base function to override for all methods
    def improveResiliency(self):
        ##for base method return original graph
        return self.graphCopy, self.caseCopy

    def getImprovementCount(self):
        return floor(self.improvedRatio * (len(self.graphCopy.es.indices) + len(self.graphCopy.vs)))

    def getImprovement(self):
        return floor(self.improvementRatio * (sum(self.graphCopy.es["c"]) + sum(self.graphCopy.vs["c"])))

# class base for ESP from random_walkers
class ESPBase(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, methodName, N, graphCopy, caseCopy, alpha, destroyMethod, H,
                 M,
                 improvedRatio,
                 improvementRatio,vMaxK=1, vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, methodName, N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK,
                            vStep)
        self.H = H
        self.M = M
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def etropyRow(self, a):
        f = np.vectorize(self.entropy)
        return -np.sum(f(a))

    def entropy(self, x):
        return x * log(x) if x != 0.0 else 0.0

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'H', 'M', 'improvementRatio', 'improvement',
                          'improvedRatio', 'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3], 'H': self.H, 'M': self.M,
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class ESPEdge(ESPBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                 improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        ESPBase.__init__(self,dataName, simName, processesCount, outputDir, 'ESP edge', N, graphCopy, caseCopy, alpha,
                         destroyMethod, H,
                         M,
                         improvedRatio, improvementRatio,vMaxK, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(self.graphCopy.es.indices), len(vs)), dtype=float)
        h = floor(self.H * self.graphCopy.average_path_length(directed=False))
        # num_cores = self.processesCount
        for vertex in vs:
            logging.log(logging.INFO, "starting walks for : " + str(vertex))
            walks = Parallel(n_jobs=self.M)(
                delayed(createRandomWalk)(self.graphCopy, vertex.index, h) for i in range(0, self.M))
            for walk in walks:
                for i, e in enumerate([row[1] for row in walk]):
                    if (e is not None and i != (len(walk) - 1)):
                        espResult[e, vertex.index] += 1.0 / self.M
                        # logging.log(logging.INFO, "finished walks for : " + str(vertex))
        results = np.apply_along_axis(self.etropyRow, 1, espResult)
        # selected for improvement edge indices
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        slacks = self.graphCopy.es(deletable=False).indices
        for index, slack in enumerate(slacks):
            results[slack] = 0.0
        selectedResults = (-results).argsort()[:improvementCount]
        logging.log(logging.INFO, "selected results: " + str(selectedResults))

        # improve c in selected edges

        for res in selectedResults:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += improvement

        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

class ESPVertex(ESPBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, H, M,
                 improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        ESPBase.__init__(self,dataName, simName, processesCount, outputDir, 'ESP vertex', N, graphCopy, caseCopy, alpha,
                         destroyMethod, H,
                         M,
                         improvedRatio, improvementRatio,vMaxK, vStep)

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})

        vs = self.graphCopy.vs
        espResult = np.zeros((len(vs), len(vs)), dtype=float)
        num_cores = self.processesCount
        for vertex in vs:
            logging.log(logging.INFO, "starting walks for : " + str(vertex))
            h = floor(self.H * self.graphCopy.average_path_length(directed=False))
            walks = Parallel(n_jobs=num_cores)(
                delayed(createRandomWalk)(self.graphCopy, vertex.index, h) for i in range(0, self.M))
            for walk in walks:
                for i, v in enumerate([row[0] for row in walk]):
                    if (v is not None and v != vertex.index):
                        espResult[v, vertex.index] += 1.0 / self.M
        # print espResult
        results = np.apply_along_axis(self.etropyRow, 1, espResult)
        # selected for improvement edge indices
        improvementCount = floor(self.improvedRatio * len(vs))
        improvement = floor(self.improvementRatio * sum(self.graphCopy.vs["c"]))
        slacks = self.graphCopy.vs(deletable=False).indices
        vMaxKIndex = self.graphCopy.vs.select(_degree=self.graphCopy.maxdegree())[0].index
        results[vMaxKIndex] = 0.0
        for index,slack in enumerate(slacks):
            results[slack] = 0.0
        selectedResults = (-results).argsort()[:improvementCount]
        logging.log(logging.INFO, "selected results: " + str(selectedResults))
        for res in selectedResults:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

class RandomEdge(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'Random edge', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        randomlyChosenEdges = np.random.choice(self.graphCopy.es.indices, improvementCount, replace=False)
        for res in randomlyChosenEdges:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy



    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class RandomVertex(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'Random vertex', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        slacks = self.graphCopy.vs(deletable=False).indices
        randomIndices = self.graphCopy.vs.indices
        vMaxKIndex = self.graphCopy.vs.select(_degree=self.graphCopy.maxdegree())[0].index
        for index ,slack in enumerate(slacks):
            randomIndices.remove(slack)
        if vMaxKIndex in randomIndices:
            randomIndices.remove(vMaxKIndex)
        randomlyChosenVertices = np.random.choice(randomIndices, improvementCount, replace=False)
        for res in randomlyChosenVertices:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class ClosenessVertex(MethodBase):
    def __init__(self, dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod,
                 improvedRatio,
                 improvementRatio, vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self, dataName, simName, processesCount, outputDir, 'Closeness vertex', N, graphCopy, caseCopy, alpha,
                    destroyMethod, vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        vs  = self.graphCopy.vs
        num_cores = self.processesCount
        harmonicClosenes = Parallel(n_jobs=num_cores,verbose=50)(
            delayed(harmonicClosenessForV)(i.index, self.graphCopy) for i in vs)
        # print espResult
        vMaxKIndex = self.graphCopy.vs.select(_degree=self.graphCopy.maxdegree())[0].index
        harmonicClosenes[vMaxKIndex] = 0.0
        slacks = self.graphCopy.vs(deletable=False).indices
        for index, slack in enumerate(slacks):
            harmonicClosenes[int(slack)] = 0.0
        chosen = (-np.array(harmonicClosenes)).argsort()[:improvementCount]
        for res in chosen:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod', 'destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod, 'destroyedItems': i[4], 'alpha': self.alpha}) for i
             in csvResult]
        outputCSVFile.close()

class ClosenessEdge(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'Closeness edge', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        es = self.graphCopy.es
        num_cores = self.processesCount
        shortestPathParticipations = Parallel(n_jobs=num_cores,verbose=50)(
            delayed(shortestPathsParticipationforE)(i.index, self.graphCopy) for i in es)
        # print espResult
        ##TODO
        slacks = self.graphCopy.es(deletable=False).indices
        for i, slack in enumerate(slacks):
            shortestPathParticipations[slack]
        chosen = (-np.array(shortestPathParticipations)).argsort()[:improvementCount]
        for res in chosen:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

    def __init__(self, dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod,
                 improvedRatio,
                 improvementRatio, vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self, dataName, simName, processesCount, outputDir, 'Closeness edge', N, graphCopy, caseCopy, alpha,
                            destroyMethod, vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

class SlackPercentageVertex(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'SlackPercentage vertex', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        slacks = self.graphCopy.vs(deletable=False).indices
        simTask = SimTask(self.methodName, 0, self.graphCopy.copy(), copyCase(self.caseCopy),vMaxK=self.vMaxK)
        simTask.updateGraphFlow(simTask.graph,simTask.case)
        vs = simTask.graph.vs
        results = np.zeros(len(vs),dtype=float)
        for i, v in enumerate(vs):
            if v["deletable"]:
                results[v.index] = (v["c"] - v["Pin"])/v["c"]
            else:
                results[v.index] = float("inf")
        for index ,slack in enumerate(slacks):
            results[slack] = float("inf")
        vMaxKIndex = self.graphCopy.vs.select(_degree=self.graphCopy.maxdegree())[0].index
        if vMaxKIndex in results:
            results[vMaxKIndex] = float("inf")
        chosen = (results).argsort()[:improvementCount]
        for res in chosen:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class SlackPercentageEdge(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'SlackPercentage vertex', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = self.getImprovementCount()
        improvement = self.getImprovement()
        slacks = self.graphCopy.es(deletable=False).indices
        simTask = SimTask(self.methodName, 0, self.graphCopy.copy(), copyCase(self.caseCopy),vMaxK=self.vMaxK)
        simTask.updateGraphFlow(simTask.graph,simTask.case)
        es = simTask.graph.es
        results = np.zeros(len(es),dtype=float)
        for i, e in enumerate(es):
            if e["deletable"]:
                results[e.index] = (e["c"] - fabs(e["Pin"]))/e["c"]
            else:
                results[e.index] = float("inf")
        for index ,slack in enumerate(slacks):
            results[slack] = float("inf")
        chosen = (results).argsort()[:improvementCount]
        for res in chosen:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()



    # take from queue
    # run SimTask
    # p
def greedySim(inputQ, outputQ,logQ):
    configureProcessLogger(logQ)
    while True:
        name = multiprocessing.current_process().name
        isAlive = multiprocessing.current_process().is_alive()
        logger = logging.getLogger()
        logger.log(logging.INFO,
                   "current simTasks queue size is %(size)d, process name %(name)s" % {
                       "size": inputQ.qsize(),
                       "name": name})
        logger.log(logging.INFO, "am alive? " + str(isAlive))
        # grab task from input queue
        simTask = None
        try:
            greedyTask = inputQ.get()
            simTask = greedyTask[0]
            element = greedyTask[1]
            elementDeletable = greedyTask[2]
            vMaxK = None
            greedyResult = 0.0
            if len(greedyTask)>3:
                vMaxK = greedyTask[3]
            if simTask is not None and elementDeletable and element is not vMaxK:
                logger.log(logging.INFO,
                           "starting %(method)s method greedySim" % {"method": simTask.method})
                simTask.runSimulation()
                result = simTask.getResult()
                logger.log(logging.INFO,
                           "finished with succes %(method)s method  with result: LCC ratio %(lcg)f , PfPd ratio %(pf)f" %
                           {"method": simTask.method, "lcg": result[1],
                            "pf": result[2]})
                greedyResult = (result[1] + result[2])/2
            outputQ.put([element,greedyResult])
        except Exception as x:
            logger.error(x)
            # result = simTask.getResult()
            if simTask is not None:
                logger.log(logging.INFO,
                           "finished with error %(method)s method with result" %
                           {"method": simTask.method})
            else:
                logger.log(logging.INFO,
                           "finished with error ")
            outputQ.put(0.0)
        finally:
            if simTask is not None:
                inputQ.task_done()

class GreedyVertex(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,greedyN,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'Greedy vertex', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio
        self.greedyN = greedyN

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = int(self.getImprovementCount())
        improvement = self.getImprovement()
        vs = self.graphCopy.vs
        vMaxKIndex = self.graphCopy.vs.select(_degree=self.graphCopy.maxdegree())[0].index
        results = []
        logQ  = multiprocessing.Queue(-1)
        simLogPath = os.path.join(self.outputDir, self.simName + "_" + strftime("%H-%M", gmtime()) + ".log")
        logListener = multiprocessing.Process(target=listener_process, args=(logQ, self.outputDir, simLogPath))
        logListener.daemon = True
        logListener.start()
        for i in range(0,improvementCount):
            greedySimTasks=multiprocessing.JoinableQueue()
            outputQ = multiprocessing.Queue()
            for j, v in enumerate(vs):
                for k in range(0, self.greedyN):
                    localGraph = self.graphCopy.copy()
                    for g, res in enumerate(results):
                        resVs = localGraph.vs.find(res)
                        resVs["c"] = resVs["c"]  + improvement
                    localVs = localGraph.vs.find(v.index)
                    localVs["c"] = localVs["c"] + improvement
                    simTask = SimTask(self.methodName, 0, localGraph, copyCase(self.caseCopy),vMaxK=self.vMaxK)
                    greedySimTasks.put([simTask,v.index,v["deletable"],vMaxKIndex])
            for j in range(self.processesCount):
                simP = multiprocessing.Process(target=greedySim,args=(greedySimTasks,outputQ,logQ))
                simP.daemon = True
                simP.start()
            greedySimTasks.join()
            greedySimTaksResult =[]
            for j in range(outputQ.qsize()):
                outputResult = outputQ.get()
                greedySimTaksResult.append(outputResult)
            npGSTR = np.array(greedySimTaksResult)
            means = np.zeros(len(vs))
            for j,v in enumerate(vs.indices):
                cut = npGSTR[npGSTR[:,0] == v][:,1]
                means[v] = np.average(cut)
            results.append(int(means.argsort()[-1]))

        for res in results:
            vertex = self.graphCopy.vs.find(int(res))
            if vertex["c"] != 0.0:
                vertex["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()

class GreedyEdge(MethodBase):
    def __init__(self,dataName, simName, processesCount, outputDir, N, graphCopy, caseCopy, alpha, destroyMethod, improvedRatio,
                 improvementRatio,greedyN,vMaxK=1,
                 vStep=None):
        MethodBase.__init__(self,dataName, simName, processesCount, outputDir, 'Greedy edge', N, graphCopy, caseCopy, alpha,
                            destroyMethod,vMaxK, vStep)
        self.improvedRatio = improvedRatio
        self.improvementRatio = improvementRatio
        self.greedyN = greedyN

    def improveResiliency(self):
        logging.log(logging.INFO,
                    "starting improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        improvementCount = int(self.getImprovementCount())
        improvement = self.getImprovement()
        slacks = self.graphCopy.es(deletable=False).indices
        simTask = SimTask(self.methodName, 0, self.graphCopy.copy(), copyCase(self.caseCopy),vMaxK=self.vMaxK)
        simTask.updateGraphFlow(simTask.graph,simTask.case)
        es = simTask.graph.es
        results = []
        logQ = multiprocessing.Queue(-1)
        simLogPath = os.path.join(self.outputDir, self.simName + "_" + strftime("%H-%M", gmtime()) + ".log")
        logListener = multiprocessing.Process(target=listener_process, args=(logQ, self.outputDir, simLogPath))
        logListener.daemon = True
        logListener.start()
        for i in range(0, improvementCount):
            greedySimTasks = multiprocessing.JoinableQueue()
            outputQ = multiprocessing.Queue()
            for j, e in enumerate(es):
                for k in range(0, self.greedyN):
                    localGraph = self.graphCopy.copy()
                    for g, res in enumerate(results):
                        resEs = localGraph.es.find(res)
                        resEs["c"] = resEs["c"] + improvement
                    localEs = localGraph.es.find(e.index)
                    localEs["c"] = localEs["c"] + improvement
                    simTask = SimTask(self.methodName, 0, localGraph, copyCase(self.caseCopy), vMaxK=self.vMaxK)
                    greedySimTasks.put([simTask, e.index, e["deletable"], None])
            for j in range(self.processesCount):
                simP = multiprocessing.Process(target=greedySim, args=(greedySimTasks, outputQ, logQ))
                simP.daemon = True
                simP.start()
            greedySimTasks.join()
            greedySimTaksResult = []
            for j in range(outputQ.qsize()):
                outputResult = outputQ.get()
                greedySimTaksResult.append(outputResult)
            npGSTR = np.array(greedySimTaksResult)
            means = np.zeros(len(es))
            for j, e in enumerate(es.indices):
                cut = npGSTR[npGSTR[:, 0] == e][:, 1]
                means[e] = np.average(cut)
            results.append(int(means.argsort()[-1]))

        for res in results:
            edge = self.graphCopy.es.find(int(res))
            if edge["c"] != 0.0:
                edge["c"] += improvement
        logging.log(logging.INFO,
                    "finishing improvement of graph resiliency for method %(method)s" % {"method": self.methodName})
        return self.graphCopy, self.caseCopy

    def serializeCSV(self, csvResult):
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        csvFileName = self.simName + "_" + self.serializationTime + ".csv"
        outputCSVFilePath = os.path.join(self.outputDir, csvFileName)
        with open(outputCSVFilePath, 'wb') as outputCSVFile:
            fieldNames = ['status', 'n', 'LCCRatio', 'PfPdRatio', 'improvementRatio', 'improvement', 'improvedRatio',
                          'improvedCount', 'method',
                          'destroyMethod','destroyedItems', 'alpha']
            writer = csv.DictWriter(outputCSVFile, delimiter=";", fieldnames=fieldNames)

            writer.writeheader()
            [writer.writerow({'status': i[0], 'n': i[1], 'LCCRatio': i[2], 'PfPdRatio': i[3],
                              'improvementRatio': self.improvementRatio, 'improvement': self.getImprovement(),
                              'improvedRatio': self.improvedRatio,
                              'improvedCount': self.getImprovementCount(),
                              'method': self.methodName,
                              'destroyMethod': self.destroyMethod,'destroyedItems':i[4], 'alpha': self.alpha}) for i in csvResult]
        outputCSVFile.close()