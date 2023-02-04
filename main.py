import pandas as pd
import NbClassifier as Nb

trainFile = input('What is the name of the CSV file for the training data set?\n')
testFile = input('What is the name of the CSV file for the testing data set?\n')
classColumnName = input('What is the class column name?\n')

trainDf = pd.read_csv(trainFile)
testDf = pd.read_csv(testFile)

X_train = trainDf.loc[:,trainDf.columns!=classColumnName]
y_train = trainDf.loc[:,trainDf.columns==classColumnName]
X_test = testDf.loc[:,testDf.columns!=classColumnName]
y_test = testDf.loc[:,testDf.columns==classColumnName]

nb = Nb.NbClassifier(0.6)

nb.fit(X_train, y_train)

# print(nb._predict(['60-69','ge40','15-19','0-2','no','2','left','left_low','no']))

print(nb.accuracy_score(X_test, y_test))


