#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:24:05 2018

@author: amine
"""


from __future__ import division
import math
import numpy as np


trainFile = "/Users/amine/.spyder/CA1data/train.data"
testFile = "/Users/amine/.spyder/CA1data/test.data"

# This function returns an numpy array with the formatted data 
# The formatted data are in a form of -1 and 1 for classVal1 and classVal 2 respectively
# Separates the weights from the label and zips them together to be used for train and test

def loadData(file, classVal1, classVal2, classVal3):
    data = []
    with open(file) as fobj:
        for line in fobj:
            temp = line.split(",")
            label = []
            for i in range(0, 4): # transform it to dynamic using range(len(x))
                label.append(float(temp[i]))
            if classVal1 in temp[-1].lower():
                label.append(1)
                data.append(label)
            elif classVal2 in temp[-1].lower() or classVal3 in temp[-1].lower():
                label.append(-1)
                data.append(label)
        
        pre_pro = np.array(data, dtype=float)
        x = pre_pro[:,0:4]
        y = pre_pro[:,-1]
        data = zip(x,y)
        
        return data
 


# This function trains the perceptron with train.data
def perceptronTrain(data):
    weight = np.array([0,0,0,0])
    trainErrors = []
    bias = 0
    success = 0
    L = 0.01
    
    errorInstance = 0
       
    for j in range(20):  # goes from 0 to 19       
        
        # shuffle rows of training data 
        np.random.shuffle(data)        
        
        for left,right in data:       
            activation = np.dot(weight, left) + bias          
            # If output was incorrect, update the weight
            if ((right * activation) <= 0):
                errorInstance += 1
                for i in left:
                    # updating the weight with L2 regularisation
                    weight = weight + np.dot(right, left) - 2 * L * weight
                    # updating the bias
                    bias += right 
            else:
                success = success + 1

                
    accuracyBis = (success / (success + errorInstance)) * 100   
    trainErrors.append(accuracyBis)

    return(weight, bias, trainErrors)
    

# This function tests the perceptron with test.data 
def PerceptronTest(weight1, bias1, weight2, bias2, weight3, bias3):

    sign = 0
    accuracy = np.array([0, 0, 0])
    failed = np.array([0, 0, 0])
    performance = np.array([0, 0, 0])
    
    dataTest1 = loadData(testFile, "class-1", "class-2", "class-3")
    dataTest2 = loadData(testFile, "class-2", "class-1", "class-3")
    dataTest3 = loadData(testFile, "class-3", "class-1", "class-2")
    
    
    for (left1, right1), (left2, right2), (left3, right3) in zip(dataTest1, dataTest2, dataTest3):
        
        tempArray = []
        labelArray = []
        
        activation1 = np.dot(weight1, left1) + bias1  
        tempArray.append(activation1)
        labelArray.append(right1)
        
        activation2 = np.dot(weight2, left2) + bias2  
        tempArray.append(activation2)
        labelArray.append(right2)
        
        activation3 = np.dot(weight3, left3) + bias3  
        tempArray.append(activation3)
        labelArray.append(right3)

        temp = np.argmax(tempArray)
              
        sign = np.sign (tempArray[temp])       
        if sign == labelArray[temp]:          
            performance[temp] = performance[temp] + 1 
            
        else:            
            failed[temp] = failed[temp] + 1                                               
            

    for i in range(len(performance)):
        if math.isnan(performance[i] / (performance[i] + failed[i])) == True:
            accuracy[i] = 0
        else:
            accuracy[i] = (performance[i] / (performance[i] + failed[i])) * 100 
            
    
    return(accuracy)
              


#================ run =========

arrayTrain1 = loadData(trainFile, "class-1", "class-2", "class-3")
arrayTrain2 = loadData(trainFile, "class-2", "class-1", "class-3")
arrayTrain3 = loadData(trainFile, "class-3", "class-1", "class-2")

#================ Mutli-class Perceptron Train ========
print "\n###### Multi-class Perceptron ##########"
print "\n###### Training ##########"

dataTrain1vAll = perceptronTrain(arrayTrain1)
per1 = dataTrain1vAll[2]
Weight1vAll = dataTrain1vAll[0]
print "\nclass 1 vs All, Training accuracy:", int(per1[0]), "%"

dataTrain2vAll = perceptronTrain(arrayTrain2)
per2 = dataTrain2vAll[2]
Weight2vAll = dataTrain2vAll[0]
print "class 2 vs All, Training accuracy:", int(per2[0]), "%"

dataTrain3vAll = perceptronTrain(arrayTrain3)
per3 = dataTrain3vAll[2]
Weight3vAll = dataTrain3vAll[0]
print "class 3 vs All, Training accuracy:", int(per3[0]), "%\n"

#===============Multi-class Perceptron Test==========
print "\n###### Testing ##########"
# dataTest o is left and 1 is right
DataTestMultiClass = PerceptronTest(dataTrain1vAll[0], dataTrain1vAll[1], dataTrain2vAll[0], dataTrain2vAll[1], dataTrain3vAll[0], dataTrain3vAll[1])
x = DataTestMultiClass

print "\nclass 1 vs all, Testing accuracy:", x[0], "%", "\nclass 2 vs all, Testing accuracy:", x[1], "%",  "\nclass 3 vs all, Testing accuracy:", x[2], "%\n\n",
print "\n\nWeights of Class 1 vs all : ", Weight1vAll
print "Weights of Class 2 vs all : ", Weight2vAll
print "Weights of Class 3 vs all : ", Weight3vAll, "\n"