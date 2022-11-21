import numpy as np

def getArchs(l1):
    combos = []
    for l1 in [l1]:
        for l2 in np.arange(3,20):
            for l3 in np.arange(0,20):
                if l3 == 0:
                    if l1*l2+l2 < 100:
                            combos.append([l2,0,0])
                    continue
                for l4 in np.arange(l3*2):
                    if l4 == 0:
                        if l1*l2+l2*l3+l3 < 100:
                            combos.append([l2,l3,0])
                        continue
                    elif l1*l2 + l2*l3 + l3*l4 + l4 < 100:
                        combos.append([l2,l3,l4])
    lcount = 0 
    for i,item in enumerate(combos):
        for j,l in enumerate(item):
            if l == 0:
                combos[i] = item[:j]
                break
            else:
                lcount +=1

    larch = np.random.choice(combos,15)
    larch
    actfun = ['tanh','sigmoid','linear','relu']
    schemas = getSchemas(actfun)

    numele = 0
    for item in larch:
        for i in item[1:]:
            numele +=1
    testnets = []
    net = []
    for item in larch:
        cat = len(item) -1
        testnets.append(np.hstack(([item]*len(schemas[cat]),schemas[cat])))
    testnets = np.squeeze(testnets)
    testform = []
    for item in testnets:
        for j in item:
            net = []
            for k in range(int(len(j)/2)):
                net.append([j[k],j[k+int(len(j)/2)],1])
            net.append([1,'linear',1])
            testform.append(net)
    
    return testform

def getSchemas(actfun):
    schema3 = []
    for l1 in range(4):
        for l2 in range(4):
            for l3 in range(4):
                schema3.append([actfun[l1],actfun[l2],actfun[l3]])
    schema2 = []
    for l1 in range(4):
        for l2 in range(4):
            schema2.append([actfun[l1],actfun[l2]])
    schema1 = []
    for l1 in range(4):
            schema1.append([actfun[l1]])
    return [schema1,schema2,schema3]

def getconns(layer,ulimit,llimit):
    prev_layer = layer[0]
    conn = 0
    for l in layer[1:]:
        conn += l*prev_layer
        prev_layer = l
    print(conn)
    if conn<ulimit and conn>llimit:
        return True
    return False

# def getNextLayer(l,depth):
#     if depth == 0:
#         return 0
#     else:
#         depth = depth -1
#         print(l)
#         return getNextLayer(l,depth -1)

# def mArch(inp,archs):
#     mArch = []
#     for arch in archs:
#         depth = len(arch)
#         l = np.arange(2,inp)
#         getNextLayer(l,depth)

#     return mArch