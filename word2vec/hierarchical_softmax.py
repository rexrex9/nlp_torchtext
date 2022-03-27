import math
from treelib import Tree,Node

class Hierarchical_Softmax():

    def __init__(self,vocabs):
        self.vocabs = vocabs
        self.tree = self.__getBinaryTreeByVocabs()
        self.leaf_path_map = self.__set_paths_to_leaves_map()
        #self.tree.show()

    def __getBinaryTreeByVocabs(self):
        deep=math.ceil(math.log2(len(self.vocabs)))
        startValue = len(self.vocabs)
        tree = Tree()
        tree.create_node(startValue, startValue)
        lastNodes = [startValue]
        startValue+=1
        for i in range(deep-1):
            parents = []
            for lastNode in lastNodes:
                tree.create_node(startValue, startValue, parent=lastNode)
                parents.append(startValue)
                startValue += 1
                tree.create_node(startValue, startValue, parent=lastNode)
                parents.append(startValue)
                startValue += 1
            lastNodes=parents
        i = 0
        for vacab in self.vocabs:
            tree.create_node(vacab,vacab,parent=lastNodes[math.floor(i/2)])
            i+=1

        return tree

    def __set_paths_to_leaves_map(self):
        leaf_path_map = {}
        lst = self.tree.paths_to_leaves()
        for l in lst:
            leaf_path_map[l[-1]]=l
        return leaf_path_map

    def isLeft(self,v,p):
        return v == self.tree.children(p)[0].identifier

    def getPathToByOneLeaf(self,leaf):
        '''
        根据叶子节点得到所有父节点以及是否走向左边的标注。
        :param leaf: 叶子节点的id
        :return: ([7, 8, 11], [1, 0, 0])
        '''
        path = self.leaf_path_map[leaf]
        islefts=[]
        for i in range(len(path)-1):
            islefts.append(int(self.isLeft(path[i+1],path[i])))
        return path[:-1],islefts

    def getPathByLeaves(self,leafs):
        paths,isLefts = [],[]
        for leaf in leafs:
            path,isLeft = self.getPathToByOneLeaf(leaf)
            paths.extend(path)
            isLefts.extend(isLeft)
        return paths,isLefts

    def getNodeNumber(self):
        return len(self.tree.nodes)