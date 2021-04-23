import numpy as np
import random
import copy

class BaseTreeNode(object):
    def __init__(self, node_type, parent=None):
        self.node_type = node_type
        self.children = []
        self.parent = parent
    
    def add_parent(self, node):
        self.parent = node

    def add_child(self, node):
        self.children.append(node)
    
    def younger_sibling(self):
        if self.parent is not None:
            my_kid_idx = self.parent.children.index(self)
            if len(self.parent.children) - my_kid_idx == 1:
                return None
            else:
                return self.parent.children[my_kid_idx + 1]
        else:
            return None
    
    def is_satisfied(self):
        return (self.node_type == 'T' or 
                all(map(lambda x: x is not None, self.children)))
    
    def pprint(self, indent=0):
        preamble = '|   ' * (indent-1) + '|-- ' * (min(1, indent))
        print(preamble + self.node_type)
        for child in self.children:
            child.pprint(indent + 1)
    
    def height(self):
        if len(self.children) == 0:
            child_max_height = 0
        else:
            child_max_height = max(c.height() for c in self.children)
        return 1 + child_max_height

class DataASTNode(BaseTreeNode):
    def __init__(self, node_type, token='', tail='', parent=None):
        super(DataASTNode, self).__init__(node_type, parent)
        self.token = token
        self.tail = tail
    
    def add_index(self, idx):
        self.node_data = idx
    
    def pprint(self, indent=0):
        preamble = '|   ' * (indent-1) + '|---' * (min(1, indent))
        print(preamble + f'* [{self.node_type}] {self.token}')
        for child in self.children:
            child.pprint(indent + 1)
    
    def vocab(self):
        my_set = [set([self.node_type])]
        child_sets = [c.vocab() for c in self.children]
        return set.union(*(child_sets + my_set))


if __name__ == '__main__':
    pass