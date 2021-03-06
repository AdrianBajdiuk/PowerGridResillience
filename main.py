import os
import sys
import json
from optparse import OptionParser
from Helper import configureBasicLogger


branchesFileName = "branches.csv"
busesFileName = "buses.csv"
generatorsFileName = "generators" \
                     ".csv"
destinationFileName = "powerGrid_graph"

currentDir = os.path.dirname(os.path.abspath(__file__))
# add libs folder to scan
libsPath = os.path.join(currentDir, "libs")
sys.path.append(libsPath)

import createGraph as cg
from Helper import copyCase
from PowerSim.ResiliencyMethods import *
from pypower.api import runpf
from runPfOptions import ppoption
import logging


def main():
    # configureBasicLogger(currentDir)

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default="data", help="folder with input data")
    parser.add_option("-c", "--config", dest="config", default="config.json", help="filename with config data")
    (options, args) = parser.parse_args()
    busesInput = os.path.join(options.input, busesFileName) if os.path.isabs(options.input) else os.path.join(
        currentDir, options.input, busesFileName)
    branchesInput = os.path.join(options.input, branchesFileName) if os.path.isabs(options.input) else os.path.join(
        currentDir, options.input, branchesFileName)
    generatorsInput = os.path.join(options.input, generatorsFileName) if os.path.isabs(options.input) else os.path.join(
        currentDir, options.input, generatorsFileName)
    destinationInput = os.path.join(options.input, destinationFileName) if os.path.isabs(
        options.input) else os.path.join(
        currentDir, options.input, busesFileName)
    result = cg.createGraphAndCase(busesInput, branchesInput, generatorsInput, destinationInput, False)
    graph = result["graph"]
    case = result["case"]

    configJSON = options.config if os.path.isabs(options.config) else os.path.join(
        currentDir, options.config)
    configDict = None

    with open(configJSON) as config_data:
        configDict = json.load(config_data)
    config_data.close()
    simTasks = []
    simN = configDict["N"]
    simAlpha = configDict["alpha"]
    simVStep = configDict["vStep"]
    simOutputDir = configDict["simOutput"] if os.path.isabs(configDict["simOutput"]) else os.path.join(currentDir,
                                                                                                       configDict[
                                                                                                           "simOutput"])
    if "simProcessorsCount" in configDict:
        simProcessorsCount  = configDict["simProcessorsCount"]
    dataName = os.path.basename(options.input)
    dataConfigFileName = os.path.join(options.input,"config.json")
    dataConfigDict = {}
    with open(dataConfigFileName) as configFile:
        dataConfigDict = json.load(configFile)
    vMaxK = dataConfigDict["vMaxK"]
    for task in configDict["simTasks"]:
        simTask = None
        destroyMethod = task["destroyMethod"]
        simName = task["simName"]
        dataName = os.path.basename(options.input)
        if task["methodName"] == "RandomEdge":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = RandomEdge(dataName,simName,simProcessorsCount,outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod,
                                 task["improvementCount"], task["improvement"],vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "RandomVertex":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = RandomVertex(dataName,simName,simProcessorsCount,outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod,
                                   task["improvementCount"], task["improvement"],vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "ESPEdge":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ESPEdge(dataName,simName,simProcessorsCount,outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod, task["H"],
                              task["M"],
                              task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "ESPVertex":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ESPVertex(dataName,simName,simProcessorsCount,outputDir, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod, task["H"],
                                task["M"],
                                task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "ClosenessVertex":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ClosenessVertex(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(), copyCase(case),
                                   simAlpha, destroyMethod,
                                   task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "ClosenessEdge":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = ClosenessEdge(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(),
                                      copyCase(case),
                                      simAlpha, destroyMethod,
                                      task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "SlackPercentageVertex":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = SlackPercentageVertex(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(),
                                    copyCase(case),
                                    simAlpha, destroyMethod,
                                    task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "SlackPercentageEdge":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = SlackPercentageEdge(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(),
                                            copyCase(case),
                                            simAlpha, destroyMethod,
                                            task["improvementCount"], task["improvement"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "GreedyVertex":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = GreedyVertex(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(),
                                          copyCase(case),
                                          simAlpha, destroyMethod,
                                          task["improvementCount"], task["improvement"],task["greedyN"], vMaxK=vMaxK, vStep=simVStep)
        elif task["methodName"] == "GreedyEdge":
            outputDir = os.path.join(simOutputDir, simName)
            simTask = GreedyEdge(dataName, simName, simProcessorsCount, outputDir, simN, graph.copy(),
                                   copyCase(case),
                                   simAlpha, destroyMethod,
                                   task["improvementCount"], task["improvement"], task["greedyN"], vMaxK=vMaxK,
                                   vStep=simVStep)
        elif task["methodName"] == "base":
            simName = "base"
            outputDir = os.path.join(simOutputDir, simName)
            simTask = MethodBase(dataName,simName,simProcessorsCount,outputDir,simName, simN, graph.copy(), copyCase(case), simAlpha, destroyMethod, vMaxK=vMaxK, vStep=simVStep)

        if simTask is not None:
            simTasks.append((simName, simTask))


    for simTask in simTasks:
        logging.log(logging.INFO, "starting %s task with data %s" % (simTask[0],simTask[1].dataName))
        simTask[1].start()
        simTask[1].join()
        logging.log(logging.INFO, "finished %s task with code %d" % (simTask[0],simTask[1].exitcode))
        simTask[1].terminate()

if __name__ == "__main__":
    main()
    # remove libs folder from scanning
    sys.path.remove(libsPath)
    print 'la'
