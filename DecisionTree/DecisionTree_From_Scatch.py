from collections import Counter
from sklearn.preprocessing import LabelEncoder

import numpy as np
class Node:
    def __init__(self,feature = None, threshold=None, left=None, right=None,*,value=None) -> None:
        # Decision Node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # leaf node
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_splits = 2, max_depth = 100, n_features =None) -> None:
        self.min_samples_splits = min_samples_splits
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth = 0):
        
        n_samples, n_feats = X.shape
        
        n_labels = len(np.unique(y))

        ## Check the stopping crieteria or base case for pure leaf
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_splits):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats,self.n_features,replace = False)
        # Demo example np.random.choice(5, 3, replace=False) = [3 2 0] = picking 3 numbers from a list with 5 elements
        # Default pick = self.n_features, 

        # Find the best split
        best_feature, best_thresh = self._best_split(X,y,feat_idxs)

        # Create Child nodes
        left_idxs, right_idxs = self._getIndexSplit(X[:,best_feature],best_thresh)

        left = self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(best_feature,best_thresh,left,right)
        

    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value   

    def _best_split(self,X,y,feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None # What you are looking for, best spilit index and feature for model to use

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx] # Get one single column by index
            thresholds = np.unique(X_column) # Get all the value in the column, save time computing repetitive values

            for thr in thresholds:
                gain = self._information_gain(y,X_column,thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
       

    def _information_gain(self,y,X_column, threshold):
        # Information Gain = Entropy before splitting - Entropy after splitting
        
        # parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs,right_idxs = self._getIndexSplit(X_column, threshold)

        # Calculate weight avg entropy of children
        n= len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l,e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        child_entrop = (n_l/n) * e_l + (n_r/n) * e_r

        infomation_gain = parent_entropy - child_entrop
        return infomation_gain
 
    def _entropy(self,y):
        # Entropy is a term that comes from physics and means a measure of disorder
        # Pure leaf has zero entropy, if sampele = 2, entropy = 1 if 0.5 for each category
        # Formula link : https://miro.medium.com/v2/resize:fit:640/format:webp/1*Mm67pN3Q7iXgdvlvnsvsJw.png
        # hist = np.bincount(y) # create list of number of occurence
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        hist = np.bincount(y_encoded.astype(int))

         
        ps = hist / len(y)
        return  -np.sum([ p * np.log(p) for p in ps if p> 0])
    

    def _getIndexSplit(self, X_column, spilt_threshold):
        left_idxs = np.argwhere(X_column <= spilt_threshold).flatten()
        right_idxs = np.argwhere(X_column > spilt_threshold).flatten()

        return left_idxs,right_idxs
    
    def predict(self,X):
        return [self._traverse_tree(x,self.root) for x in X]

    def _traverse_tree(self, x,node): # traverse is working on one single row, so index will be about the features
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        
        return self._traverse_tree(x,node.right)





        
