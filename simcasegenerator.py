import sys,os
import optparse
import time
import json
def main():
    parser = optparse.OptionParser()
    parser.add_option("-o", "--output", dest="output", default="configs",
                      help="folder where goes auto-generated configs")
    parser.add_option("-r", "--runs", dest="runs", default="runs",
                      help="folder where goes auto-generated run shells")
    parser.add_option("-cCount", "--countIP", dest="countIP", default=7, help="count of sim improvements percentages")
    parser.add_option("-iCount", "--countI", dest="countI", default=7, help="count of sim improvements ")
    parser.add_option("-n","--probes",dest="probes",default=100,help="number of probes to genereate")
    parser.add_option("-p", "--processors", dest="proc", default=20, help="number of processors to use")
    parser.add_option("-m","--mem",dest="mem",default=20000,help="required RAM")
    parser.add_option("-cStep", "--cStep", dest="cStep", default=0.05, help="percentage of improvement count step")
    parser.add_option("-iStep", "--iStep", dest="iStep", default=0.0002, help="improvement step")
    (options, args) = parser.parse_args()
    currentDir = os.path.dirname(os.path.abspath(__file__))

    #get parsed options:
    configOutput = options.output if os.path.isabs(options.output) else os.path.join(currentDir,options.output)
    runOutput = options.runs if os.path.isabs(options.runs) else os.path.join(currentDir,options.runs)
    countIP = options.countIP
    countI = options.countI
    probes = options.probes
    processors = options.processors
    cStep = options.cStep
    iStep = options.iStep
    methods = ["RandomEdge","RandomVertex","ESPEdge","ESPVertex"]

    if not os.path.exists(configOutput):
        os.makedirs(configOutput)
    if not os.path.exists(runOutput):
        os.makedirs(runOutput)

    for i in range(cStep,cStep*countIP,cStep):
        for j in range(iStep,iStep*countI,iStep):
            for m in methods:
                configName = m + "-imp%d-impC%d" % (j, i) + "-config"
                config = {"N":probes,"alpha":0.005,"vStep":50,"simProcessorsCount":processors,"simOutput": "simulations",
                          "simTasks":[{
                            "methodName": m,
                            "destroyMethod": "VMaxK",
                            "improvement": j,
                            "improvementCount": i
                        }]}
                configFileName = os.path.join(configOutput,configName+str(time.gmtime())+".json")
                with open(configFileName,'w') as configFile:
                    json.dump(config,configFile)
                    configFile.close()





if __name__ == "__main__":
    main()
    # remove libs folder from scanning
