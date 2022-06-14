import pandas as pd
from matplotlib import pyplot
from ModelForProvince import ModelForProvince
from Algorithms import linearRegressionFromSklear, linearCustomModel

listOfFiles = ["GOSP_2173_CTAB_20220613111410.csv", "PRZE_3824_CTAB_20220613124138.csv"]


class Model:
    def __init__(self, listOfFiles):
        self.__dataList = self.csvReader(listOfFiles)
        self.__provinceList = self.__dataList[0].index.values
        self.__model = None

    def csvReader(self, listFiles):
        result = list()
        for file in listFiles:
            df = pd.read_csv(file, delimiter=';').drop(columns="Kod")
            df.drop(df.columns[-1], axis=1, inplace=True)
            df = df[df.Nazwa != "POLSKA"]
            df.set_index("Nazwa", inplace=True)
            df = self.renameColumns(df)
            result.append(df)
        return result

    @staticmethod
    def renameColumns(df):
        for column in df:
            newColumn = column.split(';')[2]
            df.rename(columns={column: newColumn}, inplace=True)
        return df

    def getProvinces(self):
        return self.__provinceList

    def createProvince(self, province):
        matrixX = pd.DataFrame(index=self.__dataList[0].columns)
        matrixX["Bailout"] = self.__dataList[0].loc[province].values

        newColumn = []
        for year in matrixX.index.values:
            newColumn.append(self.__dataList[1].loc[province, year])
        matrixX["New apartments"] = newColumn
        matrixX.dropna(axis=0, inplace=True)

        matrixY = pd.DataFrame(index=matrixX.index.values)
        matrixY["New apartments"] = matrixX.pop("New apartments")

        self.__model = ModelForProvince(province, matrixX, matrixY, matrixX.index.values)

    def predictFromYear(self, year, algorithm):
        if self.__model is None:
            raise Exception("No province is modeling!")

        if algorithm == "LCM":
            return self.__model.predictFromYear(year, linearCustomModel)
        elif algorithm == "LSM":
            return self.__model.predictFromYear(year, linearRegressionFromSklear)
        else:
            raise Exception("Wrong algorithm name!")

    def predictFromData(self, bailout, algorithm):
        if self.__model is None:
            raise Exception("No province is modeling!")

        if algorithm == "LCM":
            return self.__model.predictFromData(bailout, linearCustomModel)
        elif algorithm == "LSM":
            return self.__model.predictFromData(bailout, linearRegressionFromSklear)
        else:
            raise Exception("Wrong algorithm name!")

    def predictFromTrainSize(self, trainSize, algorithm):
        if self.__model is None:
            raise Exception("No province is modeling!")

        if algorithm == "LCM":
            return self.__model.predictFromTrainSize(trainSize, linearCustomModel)
        elif algorithm == "LSM":
            return self.__model.predictFromTrainSize(trainSize, linearRegressionFromSklear)
        else:
            raise Exception("Wrong algorithm name!")

    def showGraphAmountYears(self, province):
        pyplot.plot(self.__dataList[1].columns, self.__dataList[1].loc[province].values)
        pyplot.title("Amount of new apartments in " + province)
        pyplot.xlabel("Years from " + self.__dataList[1].columns[0] + " to " + self.__dataList[1].columns[-1])
        pyplot.xticks(self.__dataList[1].columns, rotation='vertical')
        pyplot.ylabel("Amount of apartments")
        pyplot.show()

    def showGraphAmountYearsPrediction(self):
        if self.__model is None:
            raise Exception("No province is modeling!")

        data, _, _, _ = self.predictFromTrainSize(0.8, "LCM")
        lsmData, _, _, _ = self.predictFromTrainSize(0.8, "LSM")
        data["Year"] = data.index.values
        data.rename(columns={"Prediction amount": "Prediction Linear Sklearn Model"}, inplace=True)
        data["Prediction Linear Custom Model"] = lsmData["Prediction amount"].values

        pyplot.plot(data["Year"], data["Real amount"], label="Real amount of new apartments")
        pyplot.plot(data["Year"], data["Prediction Linear Sklearn Model"], label="Prediction Linear Sklearn Model")
        pyplot.plot(data["Year"], data["Prediction Linear Custom Model"], label="Prediction Linear Custom Model")

        pyplot.legend(loc="upper left")
        pyplot.title("Prediction of new apartments in " + self.__model.getProvince())
        pyplot.xlabel("Years from " + data["Year"].values[0] + " to " + data["Year"].values[-1])
        pyplot.xticks(data["Year"], rotation='vertical')
        pyplot.ylabel("Amount of apartments")
        pyplot.show()

    def showGraphAmountBailoutPrediction(self):
        if self.__model is None:
            raise Exception("No province is modeling!")

        data, _, _, _ = self.predictFromTrainSize(0.8, "LCM")
        lsmData, _, _, _ = self.predictFromTrainSize(0.8, "LSM")
        data.rename(columns={"Prediction amount": "Prediction Linear Sklearn Model"}, inplace=True)
        data["Prediction Linear Custom Model"] = lsmData["Prediction amount"].values

        xList = []
        for year in data.index.values:
            xList.append(self.__dataList[0].loc[self.__model.getProvince(), year])
        xList.sort()

        pyplot.plot(xList, data["Real amount"], label="Real amount of new apartments")
        pyplot.plot(xList, data["Prediction Linear Sklearn Model"], label="Prediction Linear Sklearn Model")
        pyplot.plot(xList, data["Prediction Linear Custom Model"], label="Prediction Linear Custom Model")

        pyplot.legend(loc="upper left")
        pyplot.title("Prediction of new apartments in " + self.__model.getProvince())
        pyplot.xlabel("Bailout [zł]")
        pyplot.xticks(xList[::4], rotation='vertical')
        pyplot.ylabel("Amount of apartments")
        pyplot.show()


if __name__ == '__main__':
    model = Model(listOfFiles)
    model.createProvince("DOLNOŚLĄSKIE")
    print(model.predictFromTrainSize(0.8, "LCM"))
    print(20* "-")
    print(model.predictFromTrainSize(0.8, "LSM"))
    print(20* "-")
    print(model.predictFromYear("2020", "LCM"))
    print(20* "-")
    print(model.predictFromYear("2020", "LSM"))
    print(20* "-")
    print(model.predictFromData(113927656, "LCM"))
    print(20* "-")
    print(model.predictFromData(113927656, "LSM"))
    print(20* "-")
    model.showGraphAmountYears("DOLNOŚLĄSKIE")
    model.showGraphAmountYearsPrediction()
    model.showGraphAmountBailoutPrediction()
