import pandas as pd
from sklearn.model_selection import train_test_split


class ModelForProvince:
    def __init__(self, province, matrixX, matrixY, years):
        self.__province = province
        self.__matrixX = matrixX
        self.__matrixY = matrixY
        self.__years = years

    def getProvince(self):
        return self.__province

    def getYears(self):
        return self.__years

    def predictFromYear(self, year, algorithm):
        X_train = self.__matrixX.drop(year)
        X_train = X_train.values.reshape(-1, 1)
        X_test = self.__matrixX.loc[year].values.reshape(-1, 1)

        Y_train = self.__matrixY.drop(year)
        Y_train = Y_train.values.reshape(-1, 1)
        Y_test = pd.DataFrame({"New apartments": self.__matrixY.loc[year].values}, index=[year])

        return algorithm(X_train, X_test, Y_train, Y_test)

    def predictFromTrainSize(self, trainSize, algorithm):
        X_train, X_test, Y_train, Y_test = train_test_split(self.__matrixX, self.__matrixY, test_size=trainSize,
                                                            random_state=42)

        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
        Y_train = Y_train.values.reshape(-1, 1)

        return algorithm(X_train, X_test, Y_train, Y_test)

    def predictFromData(self, data, algorithm):
        Y_test = pd.DataFrame({"New apartments": [0.0]}, index=[data])
        X_test = pd.DataFrame({"New apartments": [data]})
        result, _, _, _ = algorithm(self.__matrixX.values.reshape(-1, 1), X_test.values,
                                    self.__matrixY.values.reshape(-1, 1), Y_test)
        return result
