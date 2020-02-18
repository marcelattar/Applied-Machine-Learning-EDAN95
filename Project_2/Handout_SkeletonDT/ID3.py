from collections import Counter
from graphviz import Digraph
from collections import Counter
import math
import numpy as np

class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0
        self.__fixedAttributes = None
        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, attributes, data, target):
        
        # Some necesarry conversions
        target_array = np.array(target)
        
        ## This calculates the Entropy function, I(S)
        target_occurence = np.array(list(Counter(target).values()))
        target_freq = np.divide(target_occurence, np.sum(target_occurence))
        I_S_list = [-target_freq[i]*math.log2(target_freq[i]) for i in range(len(Counter(target).keys()))]
        I_S = sum(I_S_list)
        
        ## This calculates the Information Gain, G(S,A), for each attribute
        attr_keys = np.array(list(attributes.keys()), dtype=str)
        # This list will be used to check which attribute produces the biggest information Gain
        G_SA = np.array([attr_keys, np.zeros(len(attributes) , dtype=int)])
        
        idx1 = 0
        for j in attr_keys:
            
            attr_values = attributes.get(j)
            #print(attr_values)
            attr_I_S = [0]*len(attr_values)
            
            idx2 = 0
            temp_list = [0]*len(attr_values)
            for l in attr_values:
                
                #attr_data = np.array([data[i] for i in range(len(data)) if data[i][idx1]==l])
                attr_data = np.array([data[i] for i in range(len(data)) if l in str(data[i]))
                #attr_target = target_array[np.array([data[i][idx1]==l for i in range(len(data))])]
                attr_target = target_array[np.array([data[i][idx1]==l for i in range(len(data))])]                      
            
                # Returns a list with the number of times a class occurs for the given attribute data
                attr_target_occurence = np.array(list(Counter(attr_target).values()))
                #print(attr_target_occurence)
                # Returns a list with the frequency of a class occurence for the given attribute data
                attr_target_freq = np.divide(attr_target_occurence, np.sum(attr_target_occurence))
                
                #print(attr_target_occurence)
                
                attr_I_S_list = [-attr_target_freq[i]*math.log2(attr_target_freq[i]) for i in range(len(Counter(attr_target).keys()))]
                attr_I_S[idx2] = sum(attr_I_S_list)
                
                print(['attr_target_freq: ', attr_target_freq])
                print(['attr_data: ', attr_data])
                print(['attr_I_S_list: ', attr_I_S_list])
                
                temp_list[idx2] = np.sum(attr_target_occurence)
                
                idx2 += 1
            
            #print(temp_list)
            #print(attr_I_S)
            V = 0
            for k in range(len(attr_values)):
                V += temp_list[k]/len(data)*attr_I_S[k]
            #print(['attr_I_S: ', attr_I_S])
            #print(['temp_list: ', temp_list])
            #print(['length of V: ',len(attr_values)])
            print(['V: ', V])
            G_SA[1][idx1] = I_S - V
            idx1 +=1
        
        a = np.array(G_SA[1])
        a = a.astype(np.float)
        index = np.where(a == np.amax(a))
        best_attribute = G_SA[0][index]
        
        best_attribute = np.array2string(best_attribute)
        #best_attribute = str(best_attribute)
        #print(best_attribute)
        return best_attribute, I_S, G_SA
    
    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self,attributes, data, target, remaining_attr):
        I0=self.entropy(target) # Start entropy
        IG=[] # information gain, IG[0] is information gain for attribute 1 and so on
        N=len(target) # number of elements in dataset
        i=0 # Attribute number, 0 = color, 1=size...
        A=[]
        for a in attributes.keys():
            if a in remaining_attr:
                A.append(a)
                I=[]
                n=[]# number of element in each subset i.e. 3 in green and 13 in yellow -> [3,13]
                for x in attributes[a]: # e.g. y & g for color
                    S=[]
                    for k in range(len(target)):
                        if data[k][i]==x: S.append(target[k]) # Extract target value if match
                    I.append(self.entropy(S))
                    n.append(len(S))
                IG.append(I0-sum((n[j]/N)*I[j] for j in range(len(I))))
            i+=1
        return A[[l for l in range(len(IG)) if IG[l]==max(IG)][0]]
    
    def entropy(self,S):
        p1 = sum(i=='+' for i in S)
        p2 = sum(i=='-' for i in S)
        N = len(S)
        if p1 == 0 & p2 == 0:
            entr=0
        elif p1 == 0:
            entr=-((p2/N)*math.log2(p2/N))
        elif p2 == 0:
            entr=-((p1/N)*math.log2(p1/N))
        else:
            entr=-((p1/N)*math.log2(p1/N)+(p2/N)*math.log2(p2/N))
        return entr

    def countTarget(self,target):
        dict={}
        pos = sum(i=='+' for i in target)
        neg = sum(i=='-' for i in target)
        if pos>0: dict['+']=pos
        if neg>0: dict['-']=neg
        return dict


    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        self.__fixedAttributes = attributes
        # fill in something more sensible here... root should become the output of the recursive tree creation
        #root = self.new_ID3_node()
        #self.add_node_to_graph(root)
        root = self.__id3(data, target, attributes, None, attributes)
        return root
        
    # This is where the recursion algorithms takes place     
    def __id3(self, data, target, attributes, A, remaining_attr):
        root = self.new_ID3_node()
        sub_attributes = attributes.copy()
        print('NODE CREATED')
        root['nodes'] = []
        
        target_occurence = np.array(list(Counter(target).values()))
        target_class = np.unique(np.array(target))
        c = [[target_class[i], target_occurence[i]] for i in range(len(target_occurence))] # This is used for classCount
        tot_targets = np.sum(target_occurence)
        
        # This block is for determining which target is the most common one
        unique_classes = np.array(target)
        unique_classes = np.unique(target)
        #print(unique_classes)
        target_occurence = target_occurence.astype(np.float)
        index = np.where(target_occurence == np.amax(target_occurence))
        
        #If all samples (data) belong to one class <class_name>
        #if tot_targets in target_occurence:
        if len(set(target)) == 1:
            #class_name = target[0]
            root['label'] = unique_classes[index]
            
            root['samples'] = len(data)
            root['classCounts'] = c
            return root
        
        # If Attributes is empty 
        elif bool(attributes)==False:
            root['label'] = unique_classes[index]
            root['entropy'] = 0
            root['samples'] = len(data)
            root['classCounts'] = c
            return root # Return the single node tree Root, with label = most common class value in Samples.
        
        # Here's the recursion
        else:
            #root['nodes'] = []
            #print(['Previous A:', A])
            #attributes.pop(A, None) # I added this, see if it works
            print(['Attributes: ',attributes.keys()])
            #A, entropy, G_SA = self.find_split_attr(attributes, data, target)
            A = self.find_split_attr(attributes, data, target, remaining_attr)
            print(A)
            #A = A[2:-2] # This is just to clean up the string
#            print(entropy)
            root['attribute'] = A
            #root['entropy'] = entropy
            root['entropy'] = self.entropy(target)
            root['samples'] = len(data)
            root['classCounts'] = c
            
            
            
            # Getting the index (column) of the attribute in the data
            idx = np.array([1 if A == a else 0 for a in self.__fixedAttributes.keys()])
            print(idx)
            idx = np.argmax(idx)
            
            
            #del sub_attributes[A]
            sub_attributes.pop(A, None)
            
            #print(['A: ', A])
            #print(['sub_attributes: ', sub_attributes])
            #print(['Entropy :', entropy])
            #print(['G_SA : ', G_SA])
            #print(attributes)
            # 
            for v in attributes[A]:
                
                sub_idx = [data[i][idx]==v for i in range(len(data))] # Returns a boolean-list with the idx for the data that contains attr. value v
                #print(sub_idx)
                sub_sample = np.array(data)[sub_idx] # This array only contains the data with attr. value v
                sub_target = np.array(target)[sub_idx]
                
                #print(v)
                #print(sub_sample)
                #  If Samples(v) is empty, then
                if sub_sample.size == 0:
                    leaf_node = self.new_ID3_node()
                    print('NODE CREATED')
                    leaf_node['label'] = unique_classes[index]
                    leaf_node['samples'] = 0
                    self.add_node_to_graph(leaf_node, root['id'])
                    root['nodes'].append(leaf_node)
                else:
                    # Below this new branch add the subtree ID3 (Samples(vi), A, Attributes/{A})
                    new_node = self.__id3(sub_sample, sub_target, sub_attributes, A, sub_attributes)
                    self.add_node_to_graph(new_node, root['id'])
                    root['nodes'].append(new_node)
        print('----new iteration------')
        return root

    def predict(self, data, tree) :
        predicted = list()
        for dat in data:
            next_node=tree
            while next_node['label']==None:
                cur_attr=next_node['attribute']
                attr_idx=list(self.__fixedAttributes.keys()).index(cur_attr)
                value_idx=self.__fixedAttributes[cur_attr].index(dat[attr_idx])
                next_node=next_node['nodes'][value_idx]
            predicted.append(next_node['label'])
        return predicted