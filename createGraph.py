import csv
import logging
import time

import igraph
import numpy as np

from PowerGridResillience import Helper

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

plotFileNameOut = "deegres_plot"
csvDegreesFileNameOut = "degrees"
csvBetweenessFileNameOut = "betweeness"
csvClustersFileNameOut = "clusters"

maxPg = 2500
graph = None


##return {"graph":graph,"case":case,"base":base}
def createGraphAndCase(busesFileName, branchesFileName,generatorsFileName, destinationFileName, saveGraph):

    # csv files
    csvBranchesFile = open(branchesFileName)
    csvBusFile = open(busesFileName)
    csvGeneratorFile = open(generatorsFileName)

    # readers
    busesReader = csv.DictReader(csvBusFile, delimiter=";")
    branchesReader = csv.DictReader(csvBranchesFile, delimiter=";")
    generatorsReader = csv.DictReader(csvGeneratorFile, delimiter=";")

    verticesCount = 0;
    referenceGenerators = np.array([])
    buses = np.array([])
    generators = np.array([])
    branches= np.array([])

    graph = igraph.Graph().as_directed()
    # graph.add_vertex()
    i = True
    j = True
    counter=0
    for row in busesReader:
        counter=+1
        index = int(row['index'])
        # diff=index-counter
        # if diff>0:
        #     graph.add_vertices(diff)
        baseKv = int(row['baseKv'])
        Vm = float(row['Vm'])
        Va = float(row['Va'])
        Pd = float(row['Pd'])
        Qd = float(row['Qd'])
        type = int(row['type'])
        c=float(row['c'])
        # area = int(row['area'])
        isGenerator = type == 2
        if type == 3:
            if i:
                referenceGenerators = np.array([index, 0, 0, -9900, 9900, Vm, maxPg, 0])
                i = False
            else:
                referenceGenerators = np.vstack((referenceGenerators, [index, 0, 0, -9900, 9900, Vm, maxPg, 0]))

        if j:
            buses = np.array([index, type, Pd, Qd, Vm, Va, baseKv,1])
            j = False
        else:
            buses = np.vstack((buses,[index, type, Pd, Qd, Vm, Va, baseKv,1]))
        ##name = Bus_index
        graph.add_vertex(name="Bus_"+str(index),Pg=0.0, isGenerator=isGenerator, Pd=Pd,c=c,Pin=0.0,isDeleted=False)
    csvBusFile.close()
    i=True
    for row in generatorsReader:
        busIndex = int(row['busIndex'])
        Pg = float(row['Pg'])
        Qg = float(row['Qg'])
        Vg = float(row['Vg'])
        Pmin = float(row['Pmin'])
        Pmax = float(row['Pmax'])
        Qmin = float(row['Qmin'])
        Qmax = float(row['Qmax'])
        if i:
            generators = np.array([busIndex,Pg,Qg,Qmax,Qmin,Vg,Pmax,Pmin])
            i=False
        else:
            generators = np.vstack((generators,[busIndex,Pg,Qg,Qmax,Qmin,Vg,Pmax,Pmin]))


    generators = np.vstack((generators,referenceGenerators))
    csvGeneratorFile.close()

    i=True
    for row in branchesReader:
        try:
            fromBus=int(row['fromBus'])
            toBus=int(row['toBus'])
            r=float(row['r'])
            x=float(row['x'])
            b=float(row['b'])
            c=float(row['c'])
            if i:
                branches=np.array([fromBus,toBus,r,x,b])
                i=False
            else:
                branches=np.vstack((branches,[fromBus,toBus,r,x,b]))
            graph.add_edge("Bus_"+str(fromBus),"Bus_"+str(toBus),Pin=0.0,Pout=0,c=c,isDeleted=False)
        except igraph.InternalError:
            print("%d, %d" % (fromBus, toBus))

    csvBranchesFile.close()
    # directedGraph=graph.as_directed()
    # logging.log(logging.INFO,igraph.summary(directedGraph))
    if saveGraph:
        graph.write_graphml(destinationFileName + "_%d.GraphML" % (time.time()))
    return {"graph":graph,"case": Helper.createCase(buses, generators, branches), "base":{"buses":buses, "generators":generators, "branches":branches}}

# def degrees(graph, csvFileNameOut, plotFileNameOut):
#     degrees=graph.degree()
#     degreesCounts=[0]*(max(degrees))
#     fieldnames=['degree','count']
#
#     for i in degrees:
#         if(i!=0):
#             degreesCounts[i-1]=degreesCounts[i-1]+1
#
#     csvFileOut=open(csvFileNameOut+'.csv','w')
#     writer=csv.DictWriter(csvFileOut,fieldnames=fieldnames)
#     writer.writeheader()
#     count=0
#     for i in degreesCounts:
#         writer.writerow({'degree':count,'count':i})
#         count=count+1
#     csvFileOut.close()
#     x=numpy.arange(1,len(degreesCounts)+1)
#     plot.figure(1)
#     plot.subplot(221)
#     plot.plot(x,degreesCounts)
#     plot.yscale('linear')
#     plot.xscale('linear')
#     plot.title('linear')
#     plot.grid(True)
#
#     plot.subplot(222)
#     plot.plot(x,degreesCounts)
#     plot.yscale('log')
#     plot.xscale('log')
#     plot.title('log')
#     plot.grid(True)
#
#     #plot.show()
#
#     plot.savefig(plotFileNameOut+'_%d.png'%(time.time()))
# def betweeness(graph, csvFileNameOut):
#
#     betweeness=graph.betweenness()
#     fieldnames=['vertexID','betweeness']
#
#     csvFileOut=open(csvFileNameOut+'.csv','w')
#     writer=csv.DictWriter(csvFileOut,fieldnames=fieldnames,dialect='excel')
#     writer.writeheader()
#     count=0
#     for i in betweeness:
#         if(i!=0):
#             writer.writerow({'vertexID':count,'betweeness':i})
#         count=count+1
#     csvFileOut.close()
#
# def connectedComponents(graph,csvFileNameOut):
#     clustersStrong=graph.clusters()
#     print clustersStrong
#     csvFileOut=open(csvFileNameOut+'.csv','w')
#     fieldnames=['cluster','bag']
#     writer=csv.DictWriter(csvFileOut,fieldnames=fieldnames)
#
#     writer.writeheader()
#     count=0
#     for i in clustersStrong:
#         writer.writerow({'cluster':count, 'bag':i})
#         count=count+1
#     csvFileOut.close()
#
# graph=createGraph(branchesFileName, busesFileName, destinationFileName)
# degrees(graph, csvDegreesFileNameOut, plotFileNameOut)
# betweeness(graph,csvBetweenessFileNameOut)
# connectedComponents(graph,csvClustersFileNameOut)
