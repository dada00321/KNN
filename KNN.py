"""
KNN, K-Nearest Neighbor
=======================
In this case, we use heights and weights as features,
genders(M/F) as labels, to filter top K nearest neighbor users,
and predict the gender toward a specific feature.
"""
import numpy as np
    
class KNN():
    def __init__(self, k):
        self.set_k_value(k)
    
    def set_k_value(self, k):
        self.k = k
        
    def predict_gender(self, test_feature):
        train_features, train_labels = self.prepare_train_data()
        maximum, minimum, new_train_features = self.normalize_features(train_features)
        new_test_feature = self.normalize_test_feature(test_feature, maximum, minimum)
        return self.classify(new_train_features, train_labels, new_test_feature)
    
    #----------------------------------------------------
    # for training data
    def prepare_train_data(self):
        features = np.array(
                           [[180, 76],
                            [158, 43],
                            [176, 78],
                            [161, 49]]
                        )
        labels = ["Male","Female","Male","Female"]
        return features, labels
        
    def normalize_features(self, features):
        maximum = np.max(features, axis=0)
        minimum = np.min(features, axis=0)
        new_train_features = (features - minimum) / (maximum - minimum)
        '''
        print("maximum:", maximum)
        print("minmum:", minimum)
        print("features:\n", features)
        print("new features:\n", new_train_features)
        '''
        return maximum, minimum, new_train_features
    
    #----------------------------------------------------
    # for test data
    def normalize_test_feature(self, test_feature, maximum, minimum):
        return (test_feature - minimum) / (maximum - minimum)
    
    #----------------------------------------------------
    # major KNN function
    def classify(self, new_train_features, train_labels, new_test_feature):
        distance = (((new_train_features - new_test_feature)**2).sum(axis=1))**0.5
        #print(distance)
        sorted_indices = distance.argsort()
        #print(sorted_indices) # indices from "most similar" to "most unsimilar"
        
        label_count = dict()
        for i in range(self.k):
            label = train_labels[sorted_indices[i]]
            label_count.setdefault(label, 0)
            label_count[label] += 1
        predicted_gender = sorted(label_count.items(), key=lambda x:x[1], reverse=True)[0][0]  
        #print(predicted_gender)
        return predicted_gender

class KNN_demo():
    def __init__(self, k):
        self.knn = KNN(k)
        
    def demo_predict_gender(self, height, weight):
        test_feature = np.array([height, weight])
        predicted_gender = self.knn.predict_gender(test_feature)
        print('Predicted result of gender with \nheight {}, weight {} is \"{}\"\n'.format(height, weight, predicted_gender))
    
if __name__ == "__main__":
    k = 3
    knn = KNN_demo(k)
    
    height, weight = 174, 78
    knn.demo_predict_gender(height, weight)
    
    height, weight = 166, 62
    knn.demo_predict_gender(height, weight)