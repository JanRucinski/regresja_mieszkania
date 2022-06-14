from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression


def linearRegressionFromSklear(X_train, X_test, Y_train, Y_test):
    model_lin = LinearRegression()
    model_lin.fit(X_train, Y_train)
    Y_pred = model_lin.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_pred)
    map_err = mean_absolute_percentage_error(Y_test, Y_pred)
    r2sc = r2_score(Y_test, Y_pred)

    Y_test["Prediction amount"] = Y_pred
    Y_test.rename(columns={"New apartments": "Real amount"}, inplace=True)
    Y_test = Y_test.sort_index()

    return Y_test, ms_err, map_err, r2sc


def linearCustomModel(X_train, X_test, Y_train, Y_test):
    params, _ = curve_fit(func, xdata=X_train.ravel(), ydata=Y_train.ravel())
    model = CustomModel(func, params)
    Y_pred = model.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_pred)
    map_err = mean_absolute_percentage_error(Y_test, Y_pred)
    r2sc = r2_score(Y_test, Y_pred)

    Y_test["Prediction amount"] = Y_pred
    Y_test.rename(columns={"New apartments": "Real amount"}, inplace=True)
    Y_test = Y_test.sort_index()

    return Y_test, ms_err, map_err, r2sc


def func(x, a1, b):
    return a1 * x + b


class CustomModel:
    def __init__(self, pred_fun, params):
        self.pred_fun = pred_fun
        self.params = params

    def predict(self, x):
        return self.pred_fun(x.ravel(), *self.params)
