import os
import sys
import json
from optparse import OptionParser
import logging

branchesFileName = "branches.csv"
busesFileName = "buses.csv"
generatorsFileName = "generators" \
                     ".csv"
destinationFileName = "powerGrid_graph"

currentDir = os.getcwd()
# add libs folder to scan
libsPath = os.path.join(currentDir, "libs")
sys.path.append(libsPath)

import createGraph as cg
from Helper import copyCase
from PowerSim.ResiliencyMethods import MethodBase, ESPEdge, ESPVertex, RandomEdge, RandomVertex
import logging

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format("/logs", "sim"))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default="data", help="folder with input data")
    parser.add_option("-c", "--config", dest="config", default="config.json", help="filename with config data")
    (options, args) = parser.parse_args()
    busesInput = os.path.join(options.input, busesFileName) if os.path.isabs(options.input) else os.path.join(
        os.getcwd(), options.input, busesFileName)
    branchesInput = os.path.join(options.input, branchesFileName) if os.path.isabs(options.input) else os.path.join(
        os.getcwd(), options.input, branchesFileName)
    generatorsInput = os.path.join(options.input, generatorsFileName) if os.path.isabs(options.input) else os.path.join(
        os.getcwd(), options.input, generatorsFileName)
    destinationInput = os.path.join(options.input, destinationFileName) if os.path.isabs(
        options.input) else os.path.join(
        os.getcwd(), options.input, busesFileName)
    result = cg.createGraphAndCase(busesInput, branchesInput, generatorsInput, destinationInput, False)
    graph = result["graph"]
    case = result["case"]

    configJSON = options.config if os.path.isabs(options.config) else os.path.join(
        os.getcwd(), options.config)
    configDict = None

    with open(configJSON) as config_data:
        configDict = json.load(config_data)

    simTasks = []
    simN = configDict["N"]
    simAlpha = configDict["alpha"]
    simVStep = configDict["vStep"]
    simOutputDir = configDict["simOutput"] if os.path.isabs(configDict["simOutput"]) else os.path.join(os.getcwd(),
                                                                                                       configDict[
                                                                                                           "simOutput"])
    for task in configDict["simTasks"]:
        simTask = None
        destroyMethod = task["destroyMethod"]
        simName = ""
        if task["methodName"] == "RandomEdge":
            simName = "RandomEdge-imp%d-impC%d" % (task["improvement"], task["improvementCount"])
            outputDir = os.path.join(simOutputDir, simName)
            simTask = RandomEdge(outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod,
                                 task["improvementCount"], task["improvement"], simVStep)
        elif task["methodName"] == "RandomVertex":
            simName = "RandomVertex-imp%d-impC%d" % (task["improvement"], task["improvementCount"])
            outputDir = os.path.join(simOutputDir, simName)
            simTask = RandomVertex(outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod,
                                   task["improvementCount"], task["improvement"], simVStep)
        elif task["methodName"] == "ESPEdge":
            simName = "ESPEdge-imp%d-impC%d-H%d-M%d" % (
                task["improvement"], task["improvementCount"], task["H"], task["M"])
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ESPEdge(outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod, task["H"],
                              task["M"],
                              task["improvementCount"], task["improvement"], simVStep)
        elif task["methodName"] == "ESPVertex":
            simName = "ESPVertex-imp%d-impC%d-H%d-M%d" % (
                task["improvement"], task["improvementCount"], task["H"], task["M"])
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ESPVertex(outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod, task["H"],
                                task["M"],
                                task["improvementCount"], task["improvement"], simVStep)

        if simTask is not None:
            simTasks.append((simName, simTask))


    for simTask in simTasks:
        logging.log(logging.INFO, "starting %s task" % (simTask[0]))
        simTask[1].start()
        simTask[1].join()
        logging.log(logging.INFO, "finished %s task with code %d" % (simTask[0],simTask[1].exitcode))
        simTask[1].terminate()


if __name__ == "__main__":
    main()
    # remove libs folder from scanning
    sys.path.remove(libsPath)
    print 'la'
