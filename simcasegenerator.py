import sys,os
import optparse
import time
import json
def main():
    parser = optparse.OptionParser()
    parser.add_option("-s", "--start", dest="main", default="main.py",
                      help="main script")
    parser.add_option("-d", "--data", dest="data", default="data",
                      help="folder with data dirs")
    parser.add_option("-o", "--output", dest="output", default="configs",
                      help="folder where goes auto-generated configs")
    parser.add_option("-r", "--runs", dest="runs", default="runs",
                      help="folder where goes auto-generated run shells")
    parser.add_option("--countIP", dest="countIP", default=7, help="count of sim improvements percentages")
    parser.add_option("--countI", dest="countI", default=7, help="count of sim improvements ")
    parser.add_option("-n","--probes",dest="probes",default=100,help="number of probes to generate")
    parser.add_option("-p", "--processors", dest="processors", default=20, help="number of processors to use")
    parser.add_option("-m","--mem",dest="mem",default=20000,help="required RAM")
    parser.add_option("--cStep", dest="cStep", default=0.05, help="percentage of improvement count step")
    parser.add_option("--iStep", dest="iStep", default=0.0002, help="improvement step")
    (options, args) = parser.parse_args()
    currentDir = os.path.dirname(os.path.abspath(__file__))
    dataDir = options.data if os.path.isabs(options.data) else os.path.join(currentDir,options.data)
    datas = [x[0] for x in os.walk(dataDir)][1:]
    #get parsed options:
    configOutput = options.output if os.path.isabs(options.output) else os.path.join(currentDir,options.output)
    runOutput = options.runs if os.path.isabs(options.runs) else os.path.join(currentDir,options.runs)
    countIP = options.countIP
    countI = options.countI
    probes = options.probes
    processors = options.processors
    cStep = float(options.cStep)
    iStep = float(options.iStep)
    methods = ["RandomEdge","RandomVertex","ESPEdge","ESPVertex"]
    allRunScripts = []
    startingScript = options.main if os.path.isabs(options.main) else os.path.join(currentDir, options.main)
    startAllName = "start_all.sh"

    if not os.path.exists(configOutput):
        os.makedirs(configOutput)
    if not os.path.exists(runOutput):
        os.makedirs(runOutput)

    for i in range(0,countIP):
        improvedRatio = (i + 1) * cStep
        for j in range(0,countI):
            improvement = (j+1)* (iStep)
            for m in methods:
                improvementS = str(improvement).replace(".","_")
                improvedRatioS = str(improvedRatio).replace(".", "_")
                simName = m + "-imp%s-impC%s" % (improvementS, improvedRatioS)
                configName = simName
                configFileOutput = os.path.join(configOutput,simName)
                if not os.path.exists(configFileOutput):
                    os.makedirs(configFileOutput)
                simOutput = os.path.join(configFileOutput,"simulation")
                config = {"N":probes,"alpha":0.005,"vStep":50,"simProcessorsCount":processors,"simOutput": simOutput,
                          "simTasks":[{
                            "methodName": m,
                            "simName":simName,
                            "destroyMethod": "VMaxK",
                            "improvement": improvement,
                            "improvementCount": improvedRatio
                        }]}
                if m.startswith("ESP"):
                    config["simTasks"][0]["H"] = 1
                    config["simTasks"][0]["M"] = 23
                configFileName = os.path.join(configFileOutput,configName+".json")
                with open(configFileName,'w') as configFile:
                    json.dump(config,configFile)
                    configFile.close()
                for dir in datas:
                    datName = os.path.basename(dir)
                    writeSingleRunScript(startingScript, options.mem,options.processors,"short",configFileOutput,configFileName,dir, simName+"_"+datName+".sh",allRunScripts)
    with open(startAllName,'w') as startAllFile:
        for script in allRunScripts:
            startAllFile.write("." + script+"\n")
        startAllFile.close()


def writeSingleRunScript(startingScript,mem,proc, queueName,outputDir,configFilePath,dataPath,scriptName,registry,walltime="20:00:00"):
    outputFileName = os.path.join(outputDir,scriptName)
    with open(outputFileName,'w') as scriptFile:
        #
        # ##################file.sh#######
        # # !/bin/bash
        # # PBS -l walltime=01:00:10
        # # PBS -l select=5:ncpus=6:mem=1000m
        # # PBS -q main
        #
        # module load python
        #
        # python PowerGridResillience/main.py -c PowerGridResillience/config3.json
        #
        # ###############################
        scriptFile.write("##################file.sh#######\n")
        scriptFile.write("# !/bin/bash\n")
        scriptFile.write("# PBS -l walltime="+walltime+"\n")
        scriptFile.write("# PBS -l select=1:ncpus="+str(proc)+":mem="+str(mem)+"m\n")
        scriptFile.write("# PBS -q "+queueName+"\n\n")
        scriptFile.write("module load python\n\n")
        scriptFile.write("python "+startingScript+" -c "+configFilePath+"-d "+dataPath+"\n\n")
        scriptFile.write("###############################")
        scriptFile.close()
    registry.append(outputFileName)


if __name__ == "__main__":
    main()
    # remove libs folder from scanning
