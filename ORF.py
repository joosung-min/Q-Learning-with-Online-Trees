# %% [markdown]
# **Changes**
# 
# * 2020-08-17:
#     * Added temporal knowledge weighting in ORF class
# 
# * 2020-08-16: 
#     * Replaced map(lambda x: ...) with list(map(lambda x: ...))
#     * Replaced 'xrange' with 'range'
#     * Replaced '.has_key()' with 'if _ in .keys()'

# %%
import numpy as np
import random
import math
import unittest
from tqdm import tqdm
from collections import deque


# %%
def dataRange(X):
    """
    Accepts a list of lists (X) and returns the "column" ranges. e.g.
    
    X = [[8,7,3], 
         [4,1,9],
         [5,6,2]]
    dataRange(X) # returns: [ [4,8], [1,7], [2,9] ]
    """
    def col(j):
        return list(map(lambda x: x[j], X))

    k = len(X[0]) # number of columns in X
    return list(map(lambda j: [ min(col(j)), max(col(j)) ], range(k)))


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
        self.gamma = param['gamma'] if 'gamma' in param.keys() else 0
        self.numTests = param['numTests'] if 'numTests' in param.keys() else 10
        self.numClasses = param['numClasses'] if 'numClasses' in param.keys() else 0
        self.maxDepth = param['maxDepth'] if 'maxDepth' in param.keys() else 30 # This needs to be implemented to restrain the maxDepth of the tree. FIXME
        self.tree = Tree( Elem(param=param) )
        self.OOBError = 0
        self.error_sum = 0
        self.error_n = 0

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
        k = self.__poisson(1) # draw a random poisson
        if k == 0:
            self.OOBError = self.__updateOOBE(x,y)
        else:
            for _ in range(k):
                self.age += 1
                (j,depth) = self.__findLeaf(x,self.tree)
                j.elem.update(x,y)
                #if j.elem.numSamplesSeen > self.minSamples and depth < self.maxDepth: # FIXME: which is the correct approach?
                if j.elem.stats.n > self.minSamples and depth < self.maxDepth:
                    g = self.__gains(j.elem)
                    if any([ gg >= self.minGain for gg in g ]):
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
        return list(map(gain, tests))

    def __findLeaf(self, x, tree, depth=0):
        if tree.isLeaf(): 
            return (tree,depth)
        else:
            (dim,loc) = tree.elem.split()
            if x[dim] < loc:
                return self.__findLeaf(x,tree.left,depth+1)
            else:
                return self.__findLeaf(x,tree.right,depth+1)

    def __poisson(self,lam=1): # fix lamda = 1
      l = np.exp(-lam)
      def loop(k,p):
          return loop(k+1, p * random.random()) if (p > l) else k - 1
      return loop(0,1)

    def __updateOOBE(self,x,y):
        """
        Needs to be implemented
        """
        
        self.error = np.abs(y - self.predict(x))        
        OOBE = self.error/y
        return OOBE

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
        if self.__classify:
            self.counts[y] += 1
        else:
            self.sum += y
            self.ss += y*y

    def reset(self):
        self.n = None
        self.eps = None
        self.__classify = None
        if self.__classify:
            self.counts = None
        else:
            self.sum = None
            self.ss = None

    def pred(self): # gives predictions
        if self.__classify:
            return np.argmax(self.counts)
        else:
            return self.sum / (self.n+self.eps)
    
    def impurity(self):
        n = self.n + self.eps
        if self.__classify:
            return np.sum(list(map(lambda x: -x/n * np.log2(x/n + self.eps), self.counts))) # entropy
        else:
            prd = self.pred()
            return np.sqrt( self.ss/n - prd*prd ) # sd of node

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
        self.numSamplesSeen = numSamplesSeen
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
        self.numSamplesSeen += 1
        for test in self.tests:
            test.update(x,y)

    def updateSplit(self,dim,loc):
        self.splitDim, self.splitLoc = dim, loc

    def split(self):
        return (self.splitDim,self.splitLoc)


# %%
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
        self.ncores = ncores
        self.gamma = 0.05
        self.Xs = deque(maxlen=2)

    def update(self,x,y):
        """
        updates the random forest by updating each tree in the forest. As mentioned above, this is currently not implemented. Please replace 'pass' (below) by the appropriate implementation.
        """
        self.Xs.append(x) # dataset to construct new tree when one is discarded
        # self.Ys.append(y) # dataset to construct new tree when one is discarded

        if self.ncores > 1:
            # parallel updates
            pass # FIXME
        else:
            # sequential updates
            
            for tree in self.forest:
                tree.update(x,y) # update each t in ORTs
            
            # Temporal Knowledge Weighting
            
            ages = [tree.age for tree in self.forest]
            idx = [i for i, v in enumerate(ages) if v > 1/self.gamma]
            
            k = int(len(self.forest)/6) # Supposed to be the number of trees to test. Increasing this seems to increase RMSE
            if len(idx) > k:
                randomIdx = random.choices(idx, k=k) # choose a random tree among those older than 1/gamma
                OOBErrors = [tree.OOBError for tree in self.forest]
                # goodTree = np.argmin(OOBErrors) # find a tree with
                
                for ridx in randomIdx:
                    r = np.random.uniform(0, 1)
                    if OOBErrors[ridx] > r: # if a randomly chosen tree's OOBE is larger than some random r
                        # self.param['xrng'] = dataRange(self.Xs)
                        # self.param['xrng'] = [[-4.8, 4.8], [-3, 3], [-0.418, 0.418], [-3, 3]] # for CartPole-v1
                        self.forest[ridx] = ORT(self.param) # discard the tree and construct a new tree
                        # self.forest[ridx].update(x, y)
                        # self.forest[ridx] = self.forest[goodTree] # copy the good tree
                        
                

    def predict(self,x):
        """
        returns prediction (a scalar) of ORF based on input (x) which is a list of input variables

        usage: 

        x = [1,2,3]
        orf.predict(x)
        """
        preds = [tree.predict(x) for tree in self.forest]
        if self.classify: # if classification
            cls_counts = [0] * self.param['numClasses']
            for p in preds:
                cls_counts[p] += 1
            return np.argmax(cls_counts)
        else:
            return np.sum(preds) / (len(preds)*1.0)

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
        return mean(list(map(lambda ort: ort.tree.size(), self.forest)))

    def meanNumLeaves(self):
        return mean(list(map(lambda ort: ort.tree.numLeaves(), self.forest)))

    def meanMaxDepth(self):
        return mean(list(map(lambda ort: ort.tree.maxDepth(), self.forest)))

    def sdTreeSize(self):
        return sd([ort.tree.size() for ort in self.forest])

    def sdNumLEaves(self):
        return sd([ort.tree.numLeaves() for ort in self.forest])

    def sdMaxDepth(self):
        return sd([ort.tree.maxDepth() for ort in self.forest])
    
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

# %%
