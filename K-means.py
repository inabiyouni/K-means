import numpy as np
import copy
from dataInfo import dataInfo
#from createNode import createNode
import xml.etree.cElementTree as ET
from readData import readData
from sys import argv, exit
import matplotlib.pyplot as plt
import time
import os.path
from matplotlib import style
style.use("fivethirtyeight")

def getDistOfPnts(subMat):
    dict = {}
    maxRep = 0
    for key in subMat:
        if key in dict:
            dict[key] += 1
            if dict[key] > maxRep: maxRep = dict[key]
        else:
            dict[key] = 1
            if maxRep == 0: maxRep = 1
    return dict, maxRep

def initial_Cntrds(inst, max_min, rand_Init):
    if rand_Init:
        print("random initializing")
        inst.centroids = (np.random.rand(inst.K, inst.k) * (max_min[0, :] - max_min[1, :])) + max_min[1, :]
    else:
        inst.centroids = np.mean(inst.data[:,:-1], axis=0).reshape(1, inst.k)
        next_mean = np.empty((1, inst.k))
        for j in range(inst.K - 1):
            max_Dist_to_Centrs = - np.inf
            for i in range(inst.N):
                dis = (inst.data[i, :-1]).reshape(1, inst.k) - inst.centroids
                dis_to_Cntrs = np.linalg.norm(dis, axis=1)
                min_dist = dis_to_Cntrs
                if dis_to_Cntrs.size > 1:
                    dis_to_Cntrs.sort()
                    min_dist = dis_to_Cntrs[0]
                if min_dist > max_Dist_to_Centrs:
                    max_Dist_to_Centrs = min_dist
                    next_mean = (inst.data[i,:-1]).reshape(1, inst.k)
            inst.centroids = np.append(inst.centroids, next_mean, axis=0)


def normalizeData(inst):
    for i in range(0,inst.k):
        attr = inst.data[:,i].astype(float)
        try:
            inst.data[:,i] = (attr) / np.std(attr)
        except Exception as e:
            print (e)

def assignPnts(inst, disType, p):
    inst.clusters[:, 0] = -1.0
    cntUpdates = 0
    for i in range(0, inst.N):
        prevAssng = inst.clusters[i, 1]
        for j in range(0, inst.K):
            inst.dist_Cnt += 1
            # print("dist: ", ((np.power(data[i,:] - data[j,:], p)).sum()))
            if (disType in ("eucli", "city")):
                dist = ((np.power(abs(inst.data[i, :-1] - inst.centroids[j, :]), p)).sum()) ** (1.0 / p)
            elif (disType == 'fun1'):
                indx = inst.data[i, :-1] > inst.centroids[j, :]
                dist = (np.power((inst.data[i, np.append(indx,False)] - inst.centroids[j, indx]).sum(), p) + np.power(
                    (inst.centroids[j, np.invert(indx)] - inst.data[i, np.append(np.invert(indx),False)]).sum(), p)) ** (1.0 / p)
            elif (disType == 'fun2'):
                indx = inst.data[i, :-1] > inst.centroids[j, :]
                dist = (np.power((inst.data[i, np.append(indx,False)] - inst.centroids[j, indx]).sum(), p) +
                        np.power((inst.centroids[j, np.invert(indx)] - inst.data[i, np.append(np.invert(indx),False)]).sum(), p)) ** (1.0 / p)
                temp = np.linalg.norm(inst.data[i, :-1] - inst.centroids[j, :])
                temp = np.append(temp, np.linalg.norm(inst.data[i, :-1]))
                temp = np.append(temp, np.linalg.norm(inst.centroids[j, :]))
                denom = np.amax(temp, axis=0)
                dist = dist / denom.sum()
            elif (disType == 'cosine'):
                x = inst.data[i, :-1]
                y = inst.centroids[j, :]
                magX = abs((np.power(x, 2)).sum()) ** (1.0 / 2)
                magY = abs((np.power(y, 2)).sum()) ** (1.0 / 2)
                dist = 1 - sum(x[:] * y[:]) / magX / magY
            if (inst.clusters[i, 0] == -1.0 or inst.clusters[i, 0] > dist):
                inst.clusters[i, 0] = dist
                inst.clusters[i, 1] = j
        if inst.clusters[i, 1] != prevAssng:
            cntUpdates += 1
    return np.sum(np.power(inst.clusters[:,0], 2.0)), cntUpdates

def calCentroids(inst, epsl):
    for j in range(0, inst.K):
        ind = inst.clusters[:,1]
        samData = inst.data[ind == j, :-1]
        if samData != []:
            inst.centroids[j,:] = np.mean(samData, axis=0)

def updateGraph(fig, x, y, diffType, p = None):
    fig.set_xdata(x)#np.append(fig.get_xdata(), x))
    fig.set_ydata(y)#np.append(fig.get_ydata(), y))
    fig.set_label('%s %d' %(diffType, p)) if p != None else fig.set_label('%s' %(diffType))
    plt.draw()
    plt.pause(0.3)#1e-17)
    time.sleep(0.01)

def findMAxMin(inst):
    max_min = np.array(np.max(inst.data[:,:-1], axis=0)).reshape(1,inst.k)
    max_min = np.append(max_min, (np.min(inst.data[:, :-1], axis=0)).reshape(1,inst.k), axis=0)
    return max_min

def K_means(inst, distFun, epsl, p, animate, rand_Init):
    if (inst.clsDist == None):
        inst.clsDist, _ = getDistOfPnts(inst.data[:,-1])
        if (inst.K == 0): inst.K = len(inst.clsDist)
        inst.k = len(inst.data[0, :-1])
        normalizeData(inst)
        max_min = findMAxMin(inst)
        if animate:
            axes.set_xlim(max_min[1,0]-1, max_min[0,0]+1)
            axes.set_ylim(max_min[1,1]-1, max_min[0,1]+1)  # 10000)
        inst.N = len(inst.data[:, 0])
        inst.clusters = np.ones((len(inst.data[:, 0]), 2), int) * -1.0
        initial_Cntrds(inst, max_min, rand_Init)
        inst.itert = 0
        inst.dist_Cnt = 0
        print("iter | number of distance cal")
    print ('\r  {}  |     {:,}'.format(inst.itert, inst.dist_Cnt), end=""),
    inst.itert += 1
    SSN, pntUpdates = assignPnts(inst, distFun, p)
    if animate:
        for j in range(0, inst.K):
            ind = inst.clusters[:, 1]
            samData = inst.data[ind == j, :-1]
            updateGraph(eval('figN%d'% (j + 2)), samData[:, 0], samData[:, 1], "Cluster", j)
            #if j == 0: updateGraph(figN2, samData[:,0], samData[:,1], "Cluster", j)
            #elif j == 1: updateGraph(figN3, samData[:,0], samData[:,1], "Cluster", j)
        updateGraph(figN1, inst.centroids[:, 0], inst.centroids[:, 1], "Centroids")
    calCentroids(inst, epsl)
    if (pntUpdates == 0):
        inst.SSN = SSN
        return
    K_means(inst, distFun, epsl, p, animate, rand_Init)
    return

def calAccuracy(inst):
    purity = 0
    realClasses = inst.data[:,-1]
    newClasses = inst.clusters[:, 1]
    for j in range(0, inst.K):
        cluster = realClasses[newClasses == j]
        _, majority = getDistOfPnts(cluster)
        purity += majority
    print("\n\nTotal purity: %0.2f" %(purity / len(inst.data[:, 0])))

#main body of the program for loading data and
# running a recursive tree function to learn from data

#arguments setting
args = iter(argv)
next(args)
clustrNo = 0
n= 10000
k = 1001#100#
p = 2
epsl = 0.1
rept_Num = 1
animate = False#True#
rand_Init = False

inputFileTr = ""
inputClassesTr = ""
inputFileTs = ""
inputClassesTs = ""
distFun = "eucli"
anim_Str = ""
init_Str = ""
for item in args:
    if item == "-K":
        clustrNo = int(next(args))
    elif item == "-itr1":
        inputFileTr = next(args)
    elif item == "-itr2":
        inputClassesTr = next(args)
    elif item == "-its1":
        inputFileTs = next(args)
    elif item == "-its2":
        inputClassesTs = next(args)
    elif item == "-f":
        distFun = next(args)
    elif item == "-r":
        rept_Num = int(next(args))
    elif item == "-a":
        anim_Str = next(args)
    elif item == "-ri":
        init_Str = next(args)

if inputFileTr.lower() == "" or inputClassesTr == "":#or inputFileTs.lower() == "" or inputClassesTs.lower() == "" or
    print("You have NOT entered correct inputs!")
    exit()
if distFun.lower() == "city":
    p = 1
elif (distFun.lower() in ("fun1", "fun2", "cosine", "eucli", "")):
    p = 2
if anim_Str.lower() == "false": animate = False
elif anim_Str.lower() == "true": animate = True

if animate:
    #plot setting
    plt.show()
    plt.xlabel('k')
    plt.ylabel('r')
    plt.title('Clustering by K-means')
    axes = plt.gca()
    figN2, = axes.plot([], [], 'gx', markersize=4)#marker='x', linestyle='--', color='g')
    figN3, = axes.plot([], [], 'b*', markersize=4)#marker='*', linestyle='--', color='b')
    figN4, = axes.plot([], [], 'c+', markersize=4)  # marker='*', linestyle='--', color='b')
    figN5, = axes.plot([], [], 'k-', markersize=4)  # marker='*', linestyle='--', color='b')
    figN6, = axes.plot([], [], 'yo', markersize=4)  # marker='*', linestyle='--', color='b')
    figN7, = axes.plot([], [], 'mx', markersize=4)  # marker='*', linestyle='--', color='b')
    figN1, = axes.plot([], [], 'r+', markersize=20)
    ax = plt.subplot(111)
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 + 0.03, pos1.y0 + 0.02,  pos1.width / 1.0, pos1.height / 1.0]
    ax.set_position(pos2) # set a new position

if init_Str.lower() == "false": rand_Init = False
elif init_Str.lower() == "true": rand_Init = True

#inputFileTr = "X_iris.txt"#X_data1_rand_2_150.txt"#X_train.txt"#X_banknote.txt"#birch.txt"#
#inputClassesTr = "y_iris.txt"#y_data1_class_1_150.txt"#y_train.txt"#banknote.txt"#birch.txt"#
#inputFileTs = "X_test.txt"#"X_data2_rand_2_150.txt"#"
#inputClassesTs = "y_test.txt"#y_data2_class_1_150.txt"#"

if inputClassesTr != "":
    print("\nReading 1 set of data ...")
    clases = readData(inputClassesTr)
    data1 = readData(inputFileTr)
    data1 = np.append(data1, clases, axis=1)

if inputClassesTs != "":
    print("\nReading 2 set of data ...")
    clases = readData(inputClassesTs)
    data2 = readData(inputFileTs)
    data2 = np.append(data2, clases, axis=1)

if inputClassesTr != "":
    data = np.append(data1, data2, axis=0) if (inputClassesTs != "") else data1

#if (distFun == ""): distFun = "eucli"
#print("\nRunning K-means algorithm ...")
#inst = dataInfo(data)
#inst.K = clustrNo
#K_means(inst, distFun.lower(), epsl, p)

result_file = open('K-means_estimated classes_%s_%d_%s.txt' %(inputFileTr[:-4], p, distFun),'a+')
#for clustrNo in [3, 20, 100]:
#result_file.write('\n%sp = %d n = %d k = %d\n' % (distFun, p, n, clustrNo))
result_file.write("Estimate Classes:\n")
for rept in range(rept_Num):
    #if (os.path.isfile("data.npy")):
    #    data = np.load("data.npy")
    #else:
    #    print("Generating new random data ...")
    #    data = np.random.random((n, k))
    #    np.save("data", data)
    if (distFun == ""): distFun = "eucli"

    print("\nRunning K-means algorithm ...")
    inst = dataInfo(data)
    inst.K = clustrNo
    K_means(inst, distFun.lower(), epsl, p, animate, rand_Init)
    result_file.write("\nRepetition #%d\n"%(rept))
    for item in inst.clusters[:,1]:
        result_file.write("%d\n"%(item))
    #result_file.write("{} {:,}\n".format(inst.itert, inst.dist_Cnt))
    result_file.flush()
    calAccuracy(inst)
    print("iterations: %d\nnumber of distance cal: %d\nSSN: %0.2f" % (inst.itert, inst.dist_Cnt, inst.SSN))
#result_file.close()



if animate:
    plt.legend()
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()

