import createGraph as cg
from Helper import copyCase
from PowerSim.ResiliencyMethods import MethodBase, ESPEdge, ESPVertex
import os
import sys

branchesFileName = "branchesTest.csv"
busesFileName = "busesTest.csv"
generatorsFileName = "generatorsTest.csv"
destinationFileName = "powerGrid_graph"

currentDir = os.getcwd()
# add libs folder to scan
libsPath = os.path.join(currentDir, "libs")
sys.path.append(libsPath)
if __name__ == "__main__":
    result = cg.createGraphAndCase(busesFileName, branchesFileName, generatorsFileName, destinationFileName, False)
    baseMethod = MethodBase("base", "base", 4, result["graph"].copy(), copyCase(result["case"]), 0, "VMaxK", vStep=1)
    baseMethod.start()
    baseMethod.join()
    espEdge = ESPEdge("espEdge", 8, result["graph"].copy(), copyCase(result["case"]), 0, "VMaxK", 4, 6, 3, 40, vStep=2)
    espEdge.start()
    espEdge.join()
    espVertex = ESPVertex("espVertex", 8, result["graph"].copy(), copyCase(result["case"]), 0, "VMaxK", 4, 6, 3, 40,
                          vStep=2)
    espVertex.start()
    espVertex.join()

    # remove libs folder from scanning
    sys.path.remove(libsPath)
    print 'la'
