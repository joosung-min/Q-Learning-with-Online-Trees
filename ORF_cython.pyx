#!python
#cython: language_level=3, boundscheck=False, wraparound=False
#cython: 
import numpy as np
import sys
import random
import math
import unittest
from tqdm import tqdm
cimport numpy as np
cimport cython

ncores = 4
# pool = multiprocessing.Pool(processes=ncores)
# pool = ProcessPoolExecutor(max_workers=ncores)
def dataRange(X):
    """
    Accepts a list of lists (X) and returns the "column" ranges. e.g.
    
    X = [[8,7,3], 
         [4,1,9],
         [5,6,2]]
    dataRange(X) # returns: [ [4,8], [1,7], [2,9] ]
    """
    def col(j):
        return [x[j] for x in X]
        # return list(map(lambda x: x[j], X))

    k = len(X[0]) # number of columns in X
    return [[np.min(col(j)), np.max(col(j))] for j in range(k)]
    # return list(map(lambda j: [ min(col(j)), max(col(j)) ], range(k)))


# %%
class Tree: # Node object
    """
    # Tree - Binary tree class

    Example:
        from tree import Tree

        t1 = Tree(1)
        t1.draw()
        t2 = Tree(1,Tree(2),Tree(3))
        t3 = Tree(1,t2,Tree(4))
        t4 = Tree(1,t2,t3)
        t4.draw()

        t4.maxDepth()  # should be 4
        t4.size()      # should be 9
        t4.numLeaves() # should be 5

    """
    def __init__(self, elem, left=None, right=None):
        """
        Example:
        t1 = Tree(1)                # creates a tree with a root node (1) and no children
        t2 = Tree(1,Tree(2),Tree(3) # creates a tree with a root node (1) with left tree (2) and right tree(3)
        t3 = Tree(1,t2,Tree(4))     # creates a tree with a root node (1) with left subtree (t2) and right tree(4)

        =====================================================
        Note: Nodes in tree must have exactly 2 or 0 children
        =====================================================
        """
        assert((left == None and right == None) or (left != None and right != None))
        self.elem = elem
        self.left = left
        self.right = right

    def updateChildren(self,l,r):
        """
        updates the left and right children trees.  e.g.

        >>> t = Tree(1)
        >>> t.draw()

        Leaf(1)

        >>> t.updateChildren(Tree(2),Tree(3))
        
        __1__
        2   3
        """
        self.left, self.right = l,r

    def isLeaf(self):
        """
        returns a boolean. True if the Tree has no children, False otherwise
        """
        return self.left == None and self.right == None

    def size(self):
        """
        returns the number of internal nodes in tree
        """
        return 1 if self.isLeaf() else self.left.size() + self.right.size() + 1
    
    def numLeaves(self):
        """
        returns number of leaves in tree
        """
        return  1 if self.isLeaf() else self.left.numLeaves() + self.right.numLeaves()

    def maxDepth(self):
        """
        returns maximum depth of tree
        """
        return self.__md(1)

    def __md(self,s):
        return s if self.isLeaf() else max(self.left.__md(s+1),self.right.__md(s+1))

    def inOrder(self):
        """
        Returns the in-order sequence of tree. Needs to be implemented...
        """
        print("return in-order sequence of tree. needs to be implemented") # FIXME
        pass

    def preOrder(self):
        """
        Returns the pre-order sequence of tree. Needs to be implemented...
        """
        print("return pre-order sequence of tree. needs to be implemented") # FIXME
        pass

    def draw(self):
        """
        Draw the tree in a pretty way in the console. Good for smaller trees. You probably don't want to draw a very large tree...
        """
        print(self.treeString())

    def treeString(self,fun=False):
        """
        Returns a string representing the flattened tree
        """
        if fun:
            return "Leaf(" + self.elem.toString() + ")" if self.isLeaf() else "\n" + "\n".join( self.__pretty(spacing=1,fun=fun) ) + "\n"
        else:
            return "Leaf(" + str(self.elem) + ")" if self.isLeaf() else "\n" + "\n".join( self.__pretty(spacing=1,fun=fun) ) + "\n"

    def __pretty(self,spacing=3,fun=False):
        def paste(l, r): # l,r: string lists
            def elongate(ls):
                maxCol = np.max(list(map(len,ls)))
                return list(map(lambda  s: s + " "*(maxCol - len(s)) , ls))

            maxRow = np.max(list(map(len, [l,r]) ))
            tmp = list(map(lambda x: x + [""]*(maxRow-len(x)), [l,r]))
            # newL,newR = map(elongate,tmp)
            newL, newR = elongate(tmp)
            return [newL[i] + newR[i] for i in range(maxRow)]

        ps = self.elem.toString() if fun else str(self.elem)
        ls,rs = list(map(lambda x: [x.elem.toString() if fun else str(x.elem)] if x.isLeaf() else x.__pretty(spacing,fun), (self.left,self.right)))
        posL = ls[0].index(self.left.elem.toString() if fun else str(self.left.elem))
        posR = rs[0].index(self.right.elem.toString() if fun else str(self.right.elem))
        top = " "*posL + "_"*(spacing+len(ls[0])-posL) + ps + "_"*(spacing+posR) + " "*(len(rs[0])-posR)
        bottom = paste(ls, paste([" "*(spacing+len(ps))],rs)) # use reduce?
        return [top] + bottom


# %%
class ORT: # Tree object
    """
    constructor for Online Random Forest (ORT)
    The theory for ORT was developed by Amir Saffari. see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf

    Only one parameter in constructor: param
    param is a dictionary having at least the following entries:

    - minSamples       : minimum number of samples a node has to see before splitting
    - minGain          : minimum reduction in node impurity (classification or sd of node) required for splitting
    - xrng             : range of the input space (see utils.dataRange)

    Also for classification, you must set numClasses:
    - numClasses       : number of classes in response (it is assumed that the responses are integers 0,...,n-1)

    The following are optional parameters with defaults:
    - numClasses       : see above (default: 0, for regression)
    - numTests         : Number of potential split location and dimension pairs to test (defaul: 10)
    - maxDepth         : Maximum depth a tree is allowed to have. A tree stops growing branches that have depth = maxDepth (default: 30. NOTE THAT YOUR TREE WILL NOT GROW BEYOND 30 DEEP, SET maxDepth TO BE A VALUE GREATER THAN 30 IF YOU WANT LARGER TREES!!!)
    - gamma            : Trees that are of age 1/gamma may be discarded. see paper (default: 0, for no discarding of old trees). Currently not implemented.


    Examples:
        xrng = [[x0_min,x0_max], [x1_min,x1_max], [x2_min,x2_max]]
        param = {'minSamples': 5, 'minGain': .1, 'numClasses': 10, 'xrng': xrng}
        ort = ORT(param)
    """
    def __init__(self,param):
        self.param = param
        self.age = 0
        self.minSamples = param['minSamples']
        self.minGain = param['minGain']
        self.xrng = param['xrng']
        self.gamma = param['gamma'] if 'gamma' in param.keys() else 0.05
        self.numTests = param['numTests'] if 'numTests' in param.keys() else 10
        self.numClasses = param['numClasses'] if 'numClasses' in param.keys() else 0
        self.maxDepth = param['maxDepth'] if 'maxDepth' in param.keys() else 30 # This needs to be implemented to restrain the maxDepth of the tree. FIXME
        self.tree = Tree( Elem(param=param) )
        self.OOBError = 0
        # self.error_sum = 0
        # self.error_n = 0

    def draw(self):
        """
        draw a pretty online random tree. Usage:

        ort.draw()
        """
        print(self.tree.treeString(fun=True))

    def update(self,x,y):
        """
        updates the ORT

        - x : list of k covariates (k x 1)
        - y : response (scalar)

        usage: 

        ort.update(x,y)
        """
        # k = self.__poisson(1) # draw a random poisson
        
        k = np.random.poisson(lam=1)
        if k == 0:
            self.OOBError = np.abs(y-self.predict(x)) / y # for regression
        else:
            for _ in range(k):
                self.age += 1
                (j,depth) = self.__findLeaf(x,self.tree)
                j.elem.update(x,y)
                #if j.elem.numSamplesSeen > self.minSamples and depth < self.maxDepth: # FIXME: which is the correct approach?
                if j.elem.stats.n > self.minSamples and depth < self.maxDepth:
                    g = self.__gains(j.elem)
                    # print("this is g : ", g)
                    if any(g >= self.minGain):
                    # if any([ gg >= self.minGain for gg in g ]):
                        bestTest = j.elem.tests[np.argmax(g)]
                        j.elem.updateSplit(bestTest.dim,bestTest.loc)
                        j.updateChildren( Tree(Elem(self.param)), Tree(Elem(self.param)) )
                        j.left.elem.stats = bestTest.statsL
                        j.right.elem.stats = bestTest.statsR
                        j.elem.reset()

    def predict(self,x):
        """
        returns a scalar prediction based on input (x)

        - x : list of k covariates (k x 1)
        
        usage: 
        
        ort.predict(x)
        
        """
        return self.__findLeaf(x,self.tree)[0].elem.pred() # [0] returns the node, [1] returns the depth

    def __gains(self,elem):
        tests = elem.tests
        def gain(test):
            statsL, statsR = test.statsL,test.statsR
            nL,nR = statsL.n,statsR.n
            n = nL + nR + 1E-9
            lossL = 0 if nL==0 else statsL.impurity()
            lossR = 0 if nR==0 else statsR.impurity()
            g = elem.stats.impurity() - (nL/n) * lossL - (nR/n)  * lossR
            return 0 if g < 0 else g
        return np.array([gain(test) for test in tests])

    def __findLeaf(self, x, tree, depth=0):
        
        tr = tree    
        while True:
            if tr.isLeaf():
                return (tr, depth)
            else:
                (dim,loc) = tr.elem.split()
                tr = tr.left if x[dim] < loc else tr.right
                depth += 1
        
        # if tree.isLeaf(): 
        #     return (tree,depth)
        # else:
        #     (dim,loc) = tree.elem.split()
        #     return self.__findLeaf(x, tree.left, depth+1) if x[dim] < loc else self.__findLeaf(x,tree.right,depth+1)

        


            # if x[dim] < loc:
            #     return self.__findLeaf(x,tree.left,depth+1)
            # else:
            #     return self.__findLeaf(x,tree.right,depth+1)

class SuffStats:
    def __init__(self,numClasses=0,sm=0.0,ss=0.0):
        self.n = 0
        self.__classify = numClasses > 0
        self.eps = 1E-10
        if numClasses > 0:
            self.counts = [0] * numClasses
        else:
            self.sum = sm
            self.ss = ss

    def update(self,y):
        self.n += 1
        
        if self.__classify == False: # if regression
            self.sum += y
            self.ss += y**2
        
        else: # if classification
            self.counts[y] += 1
        
        # if self.__classify:
        #     self.counts[y] += 1
        # else:
        #     self.sum += y
        #     self.ss += y**2

    def reset(self):
        if self.__classify == False:
            self.sum = None
            self.ss = None
        else: 
            self.counts = None
        self.n = None
        self.eps = None
        self.__classify = None
        
        # if self.__classify:
        #     self.counts = None
        # else:
        #     self.sum = None
        #     self.ss = None

    def pred(self): # gives predictions
        return self.sum / (self.n+self.eps) if self.__classify == False else np.argmax(self.counts)        
        
        # if self.__classify:
        #     return np.argmax(self.counts)
        # else:
        #     return self.sum / (self.n+self.eps)

    def impurity(self):        
        n = self.n + self.eps
        if self.__classify == False: # if regression
            prd = self.pred()
            return np.sqrt( self.ss/n - prd**2 ) # sd of node
        else: # if classification
            return np.sum(list(map(lambda x: -x/n * np.log2(x/n + self.eps), self.counts))) # entropy

        # if self.__classify:
        #     return np.sum(list(map(lambda x: -x/n * np.log2(x/n + self.eps), self.counts))) # entropy
        # else:
        #     prd = self.pred()
        #     return np.sqrt( self.ss/n - prd*prd ) # sd of node

class Test:
    def __init__(self,dim,loc,numClasses):
        self.__classify = numClasses > 0
        self.statsL = SuffStats(numClasses=numClasses)
        self.statsR = SuffStats(numClasses=numClasses)
        self.dim = dim
        self.loc = loc

    def update(self,x,y):
        if x[self.dim] < self.loc:
            self.statsL.update(y) 
        else:
            self.statsR.update(y)

class Elem: #HERE
    def __init__(self,param,splitDim=-1,splitLoc=0,numSamplesSeen=0):
        self.xrng = param['xrng']
        self.xdim = len(param['xrng']) # number of features in x
        self.numClasses = param['numClasses'] if 'numClasses' in param.keys() else 0
        self.numTests = param['numTests'] if 'numTests' in param.keys() else 10
        self.splitDim = splitDim
        self.splitLoc = splitLoc
        # self.numSamplesSeen = numSamplesSeen
        self.stats = SuffStats(self.numClasses)
        self.tests = [ self.generateTest() for _ in range(self.numTests) ]

    def reset(self):
        self.stats = None #self.stats.reset()
        self.tests = None

    def generateTest(self):
        dim = random.randrange(self.xdim) # pick a random feature among x
        loc = random.uniform(self.xrng[dim][0],self.xrng[dim][1]) # pick a random value between x_min, x_max
        return Test(dim, loc, self.numClasses)

    def toString(self):
        return str(self.pred()) if self.splitDim == -1 else "X" + str(self.splitDim+1) + " < " + str(round(self.splitLoc,2))

    def pred(self): # gives the predicted value
        return self.stats.pred()
    
    def update(self,x,y):
        self.stats.update(y)
        # self.numSamplesSeen += 1
        for test in self.tests:
            test.update(x,y)

    def updateSplit(self,dim,loc):
        self.splitDim, self.splitLoc = dim, loc

    def split(self):
        return (self.splitDim,self.splitLoc)


# %%

# def treeUpdate(argslist):
#     ForestObject, x, y,  = argslist
#     for i in range(ForestObject.numTrees):
#         ForestObject[i].update(x, y)
    
#     if self.forest[i].age > 1/self.gamma:
#         self.a_idx.append(i)

class ORF:
    def __init__(self,param,numTrees=100,ncores=0):
        """
        Constructor for Online Random Forest. For more info: >>> help(ORT)

        One variable (param) is required to construct an ORF:
        - param          : same as that in ORT. see >>> help(ORT)
        param is a dictionary having at least the following entries:

            - minSamples : minimum number of samples a node has to see before splitting
            - minGain    : minimum reduction in node impurity (classification or sd of node) required for splitting
            - xrng       : range of the input space (see utils.dataRange)

            Also for classification, you must set numClasses:
            - numClasses : number of classes in response (it is assumed that the responses are integers 0,...,n-1)

            The following are optional parameters with defaults:
            - numClasses : see above (default: 0, for regression)
            - numTests   : Number of potential split location and dimension pairs to test (defaul: 10)
            - maxDepth   : Maximum depth a tree is allowed to have. A tree stops growing branches that have depth = maxDepth (default: 30. NOTE THAT YOUR TREE WILL NOT GROW BEYOND 30 DEEP, SET maxDepth TO BE A VALUE GREATER THAN 30 IF YOU WANT LARGER TREES!!!)
            - gamma      : Trees that are of age 1/gamma may be discarded. 

        Two parameters are optional: 
        - numTrees       : number of trees in forest (default: 100)
        - ncores         : number of cores to use. (default: 0). Currently NOT IMPLEMENTED, but  if ncores > 0, a parallel implementation will be invoked for speed gains. Preferrably using multiple threads. SO, DON'T USE THIS YET!!! See the update function below.
        
        usage:

        orf = ORF(param,numTrees=20)
        """
        self.param = param
        self.classify = 1 if 'numClasses' in param.keys() else 0
        self.numTrees = numTrees
        self.forest = [ORT(param) for _ in range(numTrees)]
        self.ncores = 4
        self.gamma = 0.05
        # self.a_idx = []

    def update(self,x,y, xrng):
        """
        updates the random forest by updating each tree in the forest. As mentioned above, this is currently not implemented. Please replace 'pass' (below) by the appropriate implementation.
        """
        
        # sequential updates    
        idx = [] # idx of trees with age > 1/gamma
        for i, tree in enumerate(self.forest):
            tree.update(x,y) # update each t in ORF
            if tree.age > 1/self.gamma:
                idx.append(i)
        
        # Temporal knowledge weighting:
        d = np.max([1, int(len(self.forest)/6)]) # d = int(len(self.forest)/6) showed some promising results in ORF_CartPole (200907)
        
        if len(idx) > d:
            randomIdx = np.random.choice(idx, size=d, replace=False) # randomly choose trees older than 1/gamma
            OOBErrors = [self.forest[i].OOBError for i in randomIdx]
            
            rr = np.random.uniform(0,1, size=len(OOBErrors)) # independently draw uniform
            didx = [i for i, e, r in zip(randomIdx, OOBErrors, rr) if e > r]
            self.param["xrng"] = xrng
            for i in didx:
                self.forest[i] = ORT(self.param) # discard the tree and construct a new tree                                      
                

    def predict(self,x):
        """
        returns prediction (a scalar) of ORF based on input (x) which is a list of input variables

        usage: 

        x = [1,2,3]
        orf.predict(x)
        """
        preds = [tree.predict(x) for tree in self.forest]
        
        if self.classify == 0: # if regression
            return np.sum(preds) / (len(preds)*1.0)
        
        else: # if classification
            cls_counts = [0] * self.param['numClasses']
            for p in preds:
                cls_counts[p] += 1
            return np.argmax(cls_counts)
        
        
        # if self.classify: # if classification
        #     cls_counts = [0] * self.param['numClasses']
        #     for p in preds:
        #         cls_counts[p] += 1
        #     return np.argmax(cls_counts)
        # else:
        #     return np.sum(preds) / (len(preds)*1.0)

    def predicts(self,X):
        """
        returns predictions (a list) of ORF based on inputs (X) which is a list of list input variables

        usage: 

        X = [ [1,2,3], [2,3,4], [3,4,5] ]
        orf.predict(X)
        """
        return [self.predict(x) for x in X]

    def predStat(self,x,f):
        """
        returns a statistic aka function (f) of the predictions of the trees in the forest given input x.

        usage:

        def mean(xs):
            return sum(xs) / float(len(xs))

        orf.predStat(x,f) # returns same thing as orf.predict(x). You would replace f by some other function (e.g. sd, quantile, etc.) to get more exotic statistics for predictions.
        """
        return f([tree.predict(x) for tree in self.forest])

    def meanTreeSize(self):
        """
        returns mean tree size of trees in forest. usage:

        orf.meanTreeSize()

        Same idea for next 5 methods (for mean and std. dev.)
        """
        return np.mean(list(map(lambda ort: ort.tree.size(), self.forest)))

    def meanNumLeaves(self):
        return np.mean(list(map(lambda ort: ort.tree.numLeaves(), self.forest)))

    def meanMaxDepth(self):
        return np.mean(list(map(lambda ort: ort.tree.maxDepth(), self.forest)))

    def sdTreeSize(self):
        return np.std([ort.tree.size() for ort in self.forest])

    def sdNumLEaves(self):
        return np.std([ort.tree.numLeaves() for ort in self.forest])

    def sdMaxDepth(self):
        return np.std([ort.tree.maxDepth() for ort in self.forest])
    
    def confusion(self,xs,ys):
        """
        creates a confusion matrix based on list of list of inputs xs, and list of responses (ys). Ideally, xs and ys are out-of-sample data.

        usage:

        orf.confusion(xs,ys)
        """
        n = self.param['numClasses']
        assert n > 1, "Confusion matrices can only be obtained for classification data." 
        preds = self.predicts(xs)
        conf = [[0] * n for _ in range(n)]
        for (y,p) in zip(ys,preds):
            conf[int(y)][p] += 1
        return conf
    
    def printConfusion(self,conf):
        """
        simply prints the confusion matrix from the previous confusion method.

        usage:
        conf = orf.confusion(xs,ys)
        orf.printConfusion(conf)
        """
        print("y/pred" + "\t" + "\t".join(map(str,range(self.param['numClasses']))))
        i = 0
        for row in conf:
            print(str(i) + "\t" + "\t".join(map(str,row)))
            i += 1

# Other functions:
def mean(xs):
    return np.sum(xs) / (len(xs)*1.0)

def sd(xs): 
    n = len(xs) *1.0
    mu = np.sum(xs) / n
    return np.sqrt( sum(list(map(lambda x: (x-mu)*(x-mu),xs))) / (n-1) )

cdef class ORF_DQN: 
    
    cdef public a_model, a_params, isFit
    cdef public int n_action, maxTrees

    def __init__(self, n_state, n_action, replay_size, ORFparams):
        self.n_action = n_action
        self.a_model = {} # to build RF for each action
        self.a_params = {a: ORFparams for a in range(n_action)}
        self.isFit = False
        self.maxTrees = ORFparams['maxTrees']
    
    cpdef predict(self, s):            
        # s: (4,) array (for cartpole)
        # preds = []
        # for a in range(self.n_action):
        #     preds.append(self.a_model[a].predict(s))
        
        preds = [self.a_model[a].predict(s) for a in range(self.n_action)]
        # print(preds)
        return preds

    def gen_epsilon_greedy_policy(self, float epsilon, int n_action):
        
        def policy_function(state):
            # state: (4,) array
            ran = "_"
            q_values =[0.0, 0.0]
            if np.random.uniform(0,1) < epsilon:
                ran = "Random"
                return([random.randint(0, n_action - 1), ran, q_values])
            else:
                if self.isFit == True:
                    ran = "Model"
                    q_values = self.predict(state) # (1,2) array
                    # print(q_values)
                else: 
                    ran = "Random_notFit"
                    return([random.randint(0, n_action - 1), ran, q_values])
                    # print("passed random.randint")
            return([np.argmax(q_values), ran, q_values])# int
        
        return policy_function


    # def expandForest(self, memory):
    #     for a in range(self.n_action):
    #         self.a_params[a]['xrng'] = dataRange([v[0] for v in memory if v[1] == a])
    #         lenFor = len(self.a_model[a].forest)
    #         for i in range(lenFor+1, self.maxTrees):
    #             self.a_model[a].forest[i] = ORF.ORT(self.a_params[a]) # build new empty trees
    

    cpdef replay(self, memory, int replay_size, float gamma, int episode):

        if len(memory) == replay_size: # Build initial Forests
            
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = dataRange([v[0] for v in memory if v[1] == a])
                self.a_model[a] = ORF(self.a_params[a]) # Fit initial RFs for each action            

        if len(memory) >= replay_size: # When the memory size exceeds the replay_size, start updating the RFs            
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state) # (, n_actions)
                q_values[action] = reward + gamma * np.max(self.predict(next_state)) if is_done == False else -1000 * reward
                
                # Update the RF for the action taken
                xrng = dataRange([v[0] for v in replay_data if v[1] == action])
                self.a_model[action].update(state, q_values[action], xrng)    
            self.isFit = True
               
        if episode == 100: # expand the number of trees at episode 100            
            # expandForest(memory)
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = dataRange([v[0] for v in memory if v[1] == a])
                lenFor = len(self.a_model[a].forest)
                for i in range(lenFor+1, self.maxTrees):
                    self.a_model[a].forest[i] = ORT(self.a_params[a]) # build new empty trees

def q_learning(env, estimator, n_episode, n_action, memory, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.95):
    
    total_reward_episode = np.zeros(n_episode)
    for episode in tqdm(range(n_episode)):
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False
        i = 0
        while not is_done:
            action, ran, pred = policy(state) # integer
            next_state, reward, is_done, _ = env.step(action)
            i += 1
            # next_state: 4x1 array (for cartpole)
            # reward: integer
            # is_done: bool (True/False)
            
            total_reward_episode[episode] += reward
            
            # ep[episode].append((i, state, ran, action))
            memory.append((state, action, next_state, reward, is_done))
            
            if is_done:
                break
            estimator.replay(memory, replay_size, gamma, episode)
            state = next_state
        epsilon = np.max([epsilon * epsilon_decay, 0.001])
        # print(epsilon)
    return total_reward_episode