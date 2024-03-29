# Q-learning with online trees - Joosung Min, Lloyd T. Elliott (2021)
#
#  * This paper utilizes online random forests as Q-function approximator for Q-learning.
#  * Construction of online random forests follows the codes written by Arthur Lui (https://github.com/luiarthur/ORFpy) which is based on the paper Online Random Forests by Saffari et. al. (2009) 
#   - Added features to the original code:
#     > (Attempts for) parallelization.
#     > temporal knowledge weighing.
#     > OOBE computation for regression.

import sys
import random
from math import exp, log
import unittest
from utils import mean, sd, argmax, log2, argmin
from collections import deque
# from multiprocessing import Process, Array

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
    return [[min(col(j)), max(col(j))] for j in range(k)]
    # return list(map(lambda j: [ min(col(j)), max(col(j)) ], range(k)))

class Tree:
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
        # print "return in-order sequence of tree. needs to be implemented" # FIXME
        pass

    def preOrder(self):
        """
        Returns the pre-order sequence of tree. Needs to be implemented...
        """
        # print "return pre-order sequence of tree. needs to be implemented" # FIXME
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
                maxCol = max(map(len,ls))
                return map(lambda  s: s + " "*(maxCol - len(s)) , ls)

            maxRow = max( map(len, [l,r]) )
            tmp = map(lambda x: x + [""]*(maxRow-len(x)), [l,r])
            newL,newR = map(elongate,tmp)
            return [newL[i] + newR[i] for i in range(maxRow)]

        ps = self.elem.toString() if fun else str(self.elem)
        ls,rs = map(lambda x: [x.elem.toString() if fun else str(x.elem)] if x.isLeaf() else x.__pretty(spacing,fun), (self.left,self.right))
        posL = ls[0].index(self.left.elem.toString() if fun else str(self.left.elem))
        posR = rs[0].index(self.right.elem.toString() if fun else str(self.right.elem))
        top = " "*posL + "_"*(spacing+len(ls[0])-posL) + ps + "_"*(spacing+posR) + " "*(len(rs[0])-posR)
        bottom = paste(ls, paste([" "*(spacing+len(ps))],rs)) # use reduce?
        return [top] + bottom


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
    - phi            : Trees that are of age 1/phi may be discarded. see paper (default: 0, for no discarding of old trees). Currently not implemented.


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
        self.phi = param['phi'] if 'phi' in param.keys() else 0.05
        self.numTests = param['numTests'] if 'numTests' in param.keys() else 10
        self.numClasses = param['numClasses'] if 'numClasses' in param.keys() else 0
        self.maxDepth = param['maxDepth'] if 'maxDepth' in param.keys() else 30 # This needs to be implemented to restrain the maxDepth of the tree. FIXME
        self.tree = Tree( Elem(param=param) )
        self.OOBError = 0
        self.error = deque(maxlen=1)
        self.g = []
        # self.error_sum = 0
        # self.error_n = 0

    def draw(self):
        """
        draw a pretty online random tree. Usage:

        ort.draw()
        """
        print(self.tree.treeString(fun=True))


    def __poisson(self,lam=1): # fix lamda = 1
      l = exp(-lam)
      def loop(k,p):
          return loop(k+1, p * random.random()) if (p > l) else k - 1
      return loop(0,1)


    def update(self,x,y):
        """
        updates ORT

        - x : list of k covariates (k x 1)
        - y : response (scalar)

        usage: 

        ort.update(x,y)
        """
        
        k = self.__poisson(1) # draw a random poisson
        
        # k = np.random.poisson(lam=1)
        
        if k == 0: # compute OOBE
            
            if self.numClasses == 0: # for regression
                self.error.append( min([ abs(  (y - self.predict(x) + 0.0001) / (y+0.0001) ) , 1])) # to make the error fall between 0 and 1
            else:
                self.error.append(y != self.predict(x)) # measure misclassification
            
            self.OOBError = mean(self.error)

            # self.OOBError = min([ abs(  (y - self.predict(x)) / (y+0.0001) ) , 1])
            # self.OOBError = 0
                
        else:
            for _ in range(k):
                self.age += 1
                (j,depth) = self.__findLeaf(x,self.tree) # find location of node j to update
                j.elem.update(x,y)
                if j.elem.numSamplesSeen > self.minSamples and depth < self.maxDepth: 
                # if j.elem.stats.n > self.minSamples and depth < self.maxDepth:
                    self.g = self.__gains(j.elem)
                    # print("this is g : ", g)
                    # if any(g >= self.minGain):
                    if any([ gg >= self.minGain for gg in self.g ]):
                        if j is None:
                            print('j is none!')
                        elif j.elem is None:
                            print('j.elem is none!')
                        elif j.elem.tests is None:
                            print('j.elem test is none!')
                        else:
                            bestTest = j.elem.tests[argmax(self.g)]
                        
                        # paramL = self.param
                        # paramR = self.param
                        # paramL['xrng'] = j.elem.xrngL
                        # paramR['xrng'] = j.elem.xrngR
                        # j.elem.updateSplit(bestTest.dim,bestTest.loc)
                        # j.updateChildren( Tree(Elem(paramL)), Tree(Elem(paramR)) )
                        # j.left.elem.stats = bestTest.statsL
                        # j.right.elem.stats = bestTest.statsR
                        # j.elem.reset()
                        
                        j.elem.updateSplit(bestTest.dim,bestTest.loc)
                        j.updateChildren( Tree(Elem(self.param)), Tree(Elem(self.param)) )
                        j.left.elem.stats = bestTest.statsL
                        j.right.elem.stats = bestTest.statsR
                        j.elem.reset()
                        self.minGain = self.minGain * 0.999
                        
    #@profile
    def predict(self,x):
        """
        returns a scalar prediction based on input (x)

        - x : list of k covariates (k x 1)
        
        usage: 
        
        ort.predict(x)
        
        """
        
        node = self.__findLeaf(x,self.tree)[0] # [0] returns the node, [1] returns the depth
        if node is None or node.elem is None:
            return 0
        else:
            return node.elem.pred()
        
        

    def __gains(self,elem):
        tests = elem.tests
        def gain(test):
            statsL, statsR = test.statsL,test.statsR
            nL,nR = statsL.n,statsR.n
            n = nL + nR + 1E-9
            lossL = 0 if nL==0 else statsL.impurity() # sd in regression
            lossR = 0 if nR==0 else statsR.impurity() # sd in regression
            
            if elem.stats is None:
                g = 0
            else:
                g = elem.stats.impurity() - (nL/n) * lossL - (nR/n)  * lossR
            
            return 0 if g < 0 else g
        if tests is None:
            return [0]
        else:
            return [gain(test) for test in tests]

    
    def __findLeaf(self, x, tree, depth=0):
        
        # tr = tree    
        while True:
            if tree.isLeaf() and tree is not None:
                return (tree, depth)
            else:
                (dim,loc) = tree.elem.split() # dim: feature x, loc: split threshold
                tree = tree.left if x[dim] < loc else tree.right
                depth += 1

class SuffStats:
    def __init__(self,numClasses=0,sm=0.0,ss=0.0, n=0):
        self.n = n
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

    def pred(self): # gives predictions(mean)
        return self.sum / (self.n+self.eps) if self.__classify == False else argmax(self.counts)        
        
        # if self.__classify:
        #     return np.argmax(self.counts)
        # else:
        #     return self.sum / (self.n+self.eps)

    def impurity(self):        
        n = self.n + self.eps
        if self.__classify == False: # if regression
            prd = self.pred()
            return ( self.ss/n - prd**2 )**(1/2) # sd of node
            
        else: # if classification
            return sum(list(map(lambda x: -x/n * log2(x/n + self.eps), self.counts))) # entropy

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

class Elem: # node elements
    def __init__(self,param,splitDim=-1,splitLoc=0,numSamplesSeen=0):
        self.xrng = param['xrng']
        self.xdim = len(param['xrng']) # number of features in x
        self.numClasses = param['numClasses'] if 'numClasses' in param.keys() else 0
        self.numTests = param['numTests'] if 'numTests' in param.keys() else 10
        self.splitDim = splitDim
        self.splitLoc = splitLoc
        self.numSamplesSeen = numSamplesSeen
        self.stats = SuffStats(self.numClasses)
        # self.tests = [ self.generateTest() for _ in range(self.numTests) ]
        self.tests = self.generateTest()
        # self.xrngL = param['xrng']
        # self.xrngR = param['xrng']
        # self.xrngL = [[0,0]] * len(param['xrng'])
        # self.xrngR = [[0,0]] * len(param['xrng'])
        # self.numUpdateL = 0
        # self.numUpdateR = 0

    def reset(self):
        self.stats = None #self.stats.reset()
        self.tests = None

    def generateTest(self):
        result = []
        for i in range(self.xdim):
            dim = i

            # dim = random.randrange(self.xdim) # pick a random feature among x
            if str(self.xrng[dim][0])[::-1] == 1 and str(self.xrng[dim][1])[::-1] == 1: # if the dimension is categorical or integer
                loc = random.randint(self.xrng[dim][0], self.xrng[dim][1]) # choose an integer between x_min, x_max
            else:
                loc = random.uniform(self.xrng[dim][0],self.xrng[dim][1]) # otherwise pick a random real value between x_min, x_max
            result.append(Test(dim, loc, self.numClasses))
        
        # return Test(dim, loc, self.numClasses)
        return result

    def toString(self):
        return str(self.pred()) if self.splitDim == -1 else "X" + str(self.splitDim+1) + " < " + str(round(self.splitLoc,2))

    def pred(self): # gives the predicted value
        return self.stats.pred()
    
    def update(self,x,y):
        self.stats.update(y)
        self.numSamplesSeen += 1
        
        for test in self.tests:
            # if x[test.dim] < test.loc:
            #     test.statsL.update(y)
            #     self.numUpdateL += 1
            #     if self.numUpdateL == 1:
            #         self.xrngL = [[x[i], x[i]] for i in range(len(x))] # initial xrng for L child node
            #     for j in range(len(x)):
            #         if x[j] < min(self.xrngL[j]):
            #             self.xrngL[j][0] = x[j]
            #         if x[j] > max(self.xrngL[j]):
            #             self.xrngL[j][1] = x[j]
            # else:
            #     test.statsR.update(y)
            #     self.numUpdateR += 1
            #     if self.numUpdateR == 1:
            #         self.xrngR = [[x[i], x[i]] for i in range(len(x))] # initial xrng for R child node
            #     for j in range(len(x)):
            #         if x[j] < min(self.xrngR[j]):
            #             self.xrngR[j][0] = x[j]
            #         if x[j] > max(self.xrngR[j]):
            #             self.xrngR[j][1] = x[j]
            test.update(x,y)

    def updateSplit(self,dim,loc):
        self.splitDim, self.splitLoc = dim, loc

    def split(self):
        return (self.splitDim,self.splitLoc)


class ORF:
    def __init__(self,param,numTrees=100,ncores=0, discard_freq = 1000):
        """
        # online random forest construction: original source code from https://github.com/luiarthur/ORFpy (Arthur Lui)
        Constructor for Online Random Forest. 

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
            - phi      : Trees that are of age 1/phi may be discarded. 

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
        self.phi = param['phi'] if 'phi' in param.keys() else 0.05
        self.best_tree = 0
        self.discard_n = 0
        self.tkw_n = 0
        self.discard_rate = 0.0
        self.discard_freq = discard_freq

    def expandTrees(self, maxTrees): # add new trees that are replicates of the best tree in the forest
        bestTree_idx = argmin([tree.OOBError for tree in self.forest])
        for _ in range(maxTrees- self.numTrees):
            self.forest.append(self.forest[bestTree_idx])
        self.numTrees = len(self.forest)

    def update(self,x,y):
        """
        updates the random forest by updating each tree in the forest. As mentioned above, this is currently not implemented. 
        Please replace 'pass' (below) by the appropriate implementation.
        """
        
        for tree in self.forest:
            tree.update(x,y) # update each t in ORF
        
        # Temporal knowledge weighting:
        
        idx = [i for i, tree in enumerate(self.forest) if tree.age > 1/self.phi] # idx of trees with age > 1/phi
        if len(idx) >= 1:
            self.tkw_n += 1
            randomIdx = random.sample(idx, k=1)[0]
            rr = random.uniform(0,self.discard_freq) # decrease the chance of discarding trees
            #rr = 999 # do not discard trees
            if self.forest[randomIdx].OOBError > rr:
                self.discard_n += 1
                self.forest[randomIdx] = ORT(self.param)
    
            self.discard_rate = self.discard_n / self.tkw_n                

        # d = 1 # number of trees to assess

        # if len(idx) > d:
        #     randomIdx = random.sample(idx, k=d) # randomly choose trees older than 1/phi
        #     OOBErrors = [self.forest[i].OOBError for i in randomIdx]
        #     tot_OOBErrors = [self.forest[j].OOBError for j in range(len(self.forest))]
        #     self.best_tree = self.forest[argmin(tot_OOBErrors)]

        #     rr = [random.uniform(0,1) for _ in range(len(OOBErrors))] # independently draw uniform
        #     didx = [i for i, e, r in zip(randomIdx, OOBErrors, rr) if e > r]
            
        #     self.param["xrng"] = xrng
            
        #     for i in didx:
        #         self.discard_n += 1
        #         print("discard tree", i, self.forest[i].age, self.forest[i].OOBError)
        #         self.forest[i] = ORT(self.param) # discard the tree and construct a new tree

    
    def predict(self,x):
        """
        returns prediction (a scalar) of ORF based on input (x) which is a list of input variables

        usage: 

        x = [1,2,3]
        orf.predict(x)
        """
        preds = [tree.predict(x) for tree in self.forest]
        

        if self.classify == 0: # if regression
            return sum(preds) / (len(preds)*1.0)
        
        else: # if classification
            cls_counts = [0] * self.param['numClasses']
            for p in preds:
                cls_counts[p] += 1
            return argmax(cls_counts)
        
        
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
