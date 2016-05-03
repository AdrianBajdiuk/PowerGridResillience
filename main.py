import createGraph as cg
from PowerGridResillience.Helper import copyCase
from PowerSim.ResiliencyMethods import MethodBase,ESPEdge,ESPVertex

branchesFileName = "branchesTest.csv"
busesFileName = "busesTest.csv"
generatorsFileName = "generatorsTest.csv"
destinationFileName = "powerGrid_graph"

if __name__=="__main__":
    result=cg.createGraphAndCase(busesFileName,branchesFileName,generatorsFileName,destinationFileName,False)
    baseMethod = MethodBase("base","base",4,result["graph"].copy(),copyCase(result["case"]),0,"VMaxK",vStep=1)
    baseMethod.start()
    baseMethod.join()
    espEdge = ESPEdge("espEdge",8,result["graph"].copy(),copyCase(result["case"]),0,"VMaxK",4,6,3,40,vStep=2)
    espEdge.start()
    espEdge.join()
    espVertex = ESPVertex("espVertex",8,result["graph"].copy(),copyCase(result["case"]),0,"VMaxK",4,6,3,40,vStep=2)
    espVertex.start()
    espVertex.join()


    # simThread = SimThread("sse",1,result["graph"].copy(),copyCase(result["case"]),0)
    # simThread.start()
    # simThread.join()

    #simThread.getResult().vs.select(lambda vertex: vertex["Pin"] == max(simThread.getResult().vs.select()["Pin"]))["name"]
    # toDelVerticeIndex = simThread.getResult().vs.select(_degree=simThread.getResult().maxdegree())[0].index
    # graph,case=deleteVertice(toDelVerticeIndex,simThread.getResult(),result["case"])
    # simThread = SimThread("sse2",2,graph.copy(),copyCase(case),toDelVerticeIndex)
    # simThread.start()
    # simThread.join()
    # resultSolver=runpf(result["case"],ppopt=ppoption())
    print 'la'

