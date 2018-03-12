#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27, 2018

@author: amine
"""
from __future__ import division
import numpy as np


trainFile = "/Users/amine/.spyder/CA1data/train.data"
testFile = "/Users/amine/.spyder/CA1data/test.data"

# This function returns an numpy array with the formatted data 
# The formatted data are in a form of -1 and 1 for classVal1 and classVal 2 respectively
# Separates the weights from the label and zips them together to be used for train and test
def loadData(file, classVal1, classVal2):
    data = []
    with open(file) as fobj:
        for line in fobj:
            temp = line.split(",")
            label = []
            for i in range(0, 4): # transform it to dynamic using range(len(x))
                label.append(float(temp[i]))
            if classVal1 in temp[-1]:
                label.append(1)
                data.append(label)
            elif classVal2 in temp[-1]:
                label.append(-1)
                data.append(label)
        
        pre_pro = np.array(data, dtype=float)
        x = pre_pro[:,0:4]
        y = pre_pro[:,-1]
        data = zip(x,y)
        
        return data
 

# This function trains the perceptron with train.data
def perceptronTrain(dataToTrain):
    weight = np.array([0,0,0,0])
    trainAccuracy = []
    bias = 0
    errorInstance = 0
   
    success = 0
    
    for j in range(20):  # goes from 0 to 19       

        # shuffle rows of training data 
        np.random.shuffle(dataToTrain)
        
        for left,right in dataToTrain:       
            activation = np.dot(weight, left) + bias
            
            # If output was incorrect, update the weight
            if ((right * activation) <= 0):
                errorInstance += 1
                
                for i in left:
                    weight = weight + np.dot(right, left)
                    # updating the bias
                    bias += right 
            else:
                success = success + 1
                    
    accuracyBis = (success / (success + errorInstance)) * 100   
    trainAccuracy.append(accuracyBis)

    return(weight, bias, trainAccuracy)


# This function tests the perceptron with test.data and the returned values of train
def PerceptronTest(dataTest, weight, bias):
    sign = 0
    failed = 0
    accuracy = 0
    performance = 0
    
    for left, right in dataTest:
        activation = np.dot(weight, left) + bias  
        sign = np.sign(activation)   
        
        if sign == right:          
            performance = performance + 1 
            
        else:            
            failed = failed + 1                                               

    accuracy = (performance / (performance + failed)) * 100   
    
    return (accuracy)




#=============== Perceptron Train class-1 vs class-2 ============    
dataTrain = loadData(trainFile, "class-1", "class-2")
Train1 = perceptronTrain(dataTrain)
per1 = Train1[2]
Weights = Train1[0]
#=============== Perceptron Train for class-1 vs class-3=========   
dataTrain = loadData(trainFile, "class-1", "class-3")
Train2 = perceptronTrain(dataTrain)
per2 = Train2[2]

#=============== Perceptron Train for class-2 vs class-3=========
dataTrain = loadData(trainFile, "class-2", "class-3")
Train3 = perceptronTrain(dataTrain)
per3 = Train3[2]



#================= Perceptron Test class-1 vs class-2 ==============
dataTest = loadData(testFile, "class-1", "class-2")
Test1 = PerceptronTest(dataTest, Train1[0], Train1[1])


#================= Perceptron Test for class-1 vs class-3===========
dataTest = loadData(testFile, "class-1", "class-3")
Test2 = PerceptronTest(dataTest, Train2[0], Train2[1])


#================= Perceptron Test for class-2 vs class-3 ==========
dataTest = loadData(testFile, "class-2", "class-3")
Test3 = PerceptronTest(dataTest, Train3[0], Train3[1])


print "\n###### Binary Perceptron ##########"

print "\n###### Training ##########\n"
print "class 1 vs class 2, Training accuracy:", int(per1[0]), "%"
print "class 1 vs class 3, Training accuracy:", int(per2[0]), "%"
print "class 2 vs class 3, Training accuracy:", int(per3[0]), "%\n\n"


print "###### Testing ##########\n"
print "class 1 vs class 2, Testing accuracy:",int(Test1), "%"
print "class 1 vs class 3, Testing accuracy:",int(Test2), "%"
print "class 2 vs class 3, Testing accuracy:",int(Test3), "%"

print "\n\nThe weights of Class 1 vs Class 2 are:\n ", Weights
    