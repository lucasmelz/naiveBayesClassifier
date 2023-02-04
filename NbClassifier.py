import pandas as pd

class NbClassifier:
    def __init__(self, alpha = None):
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def fit(self, x, y):
        self.Xtrain = x
        self.ytrain = y

        # getting class column name and class possible values
        classColumnName = y.keys().tolist()[0]
        classValues = y[classColumnName].unique()
        self.classColumnName = classColumnName
        self.classValues = classValues

        # getting name of attributes
        attributes = x.keys().tolist()
        self.attributes = attributes 

        # joining attributes with class
        df = x.join(y)

        # getting the number of values per attribute
        self.numberOfValuesPerAttribute = {}
        for attr in attributes:
            self.numberOfValuesPerAttribute[attr] = x[attr].nunique()

        # dictionary of conditional probabilities
        self.conditionalProbabilities = {}
        
        # probability of belonging to a single class
        self.pClass = {}

        # number of occurrences of a class
        self.nClassOccurrences = {}

        numOfAttrPerVal = self.numberOfValuesPerAttribute
        alpha = self.alpha
        
        # creating our probabilistic model

        for classValue in classValues:
            self.nClassOccurrences[classValue] = df[classColumnName].value_counts()[classValue]
            self.pClass[classValue] = (self.nClassOccurrences[classValue] + alpha) / (len(df.index) + alpha*len(classValues))
            dfClass = df[df[classColumnName]==classValue]
            for attr in attributes:
                for attrValue in df[attr].unique().tolist():
                    n = len(dfClass[dfClass[attr]==attrValue])
                    self.conditionalProbabilities[(classValue, attr, attrValue)] = (n + alpha) / (len(dfClass) + alpha*numOfAttrPerVal[attr])
    

    def predict(self, x):
        listOfPredictions = []
        for i in range(len(x)):
            currPrediction = self._predict(x.iloc[[i]].values.flatten().tolist())
            listOfPredictions.append(currPrediction)
        return listOfPredictions
    
    def _predict(self, row):
        classValues = self.classValues
        attributes = self.attributes
        conditionalProbabilities = self.conditionalProbabilities
        numOfValPerAttr = self.numberOfValuesPerAttribute
        alpha = self.alpha

        p = float('-inf')
        predictedClass = 'undefined'

        for classValue in classValues:
            pClass = self.pClass[classValue]
            for i in range(len(attributes)):
                if (classValue, attributes[i], row[i]) in conditionalProbabilities:
                    pClass = pClass * conditionalProbabilities[(classValue, attributes[i], row[i])]
                else:
                    nvals = float(numOfValPerAttr[attributes[i]])
                    pClass = pClass * alpha/(self.nClassOccurrences[classValue] + nvals*alpha)

            if pClass > p:
                predictedClass = classValue
                p = pClass

        return predictedClass

    def accuracy_score(self, x, y):
        yList = y.values.flatten().tolist()
        prediction = self.predict(x)

        correctPredictions=0

        for i in range(len(yList)):
            if yList[i]==prediction[i]:
                correctPredictions+=1

        return correctPredictions/len(yList)
            





    




