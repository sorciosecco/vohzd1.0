
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut as loo
from sklearn.cross_decomposition import PLSRegression

from core import settings

def leave_one_out(X, Y, M):
    Yp=[]
    for train_index, test_index in loo().split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        Yp.append(M.predict(X_test).tolist()[0])
    Y_pred = [ y[0] for y in Yp ]
    q2, sdep = r2_score(Y, Y_pred), math.sqrt(mean_squared_error(Y, Y_pred))
    return q2, sdep

def get_ml_predictions():
    pass

def get_pls_predictions(x_train, x_test, y_train, latent_variables):
    
    model = PLSRegression(n_components=latent_variables, scale=False)
    
    model.fit(x_train, y_train)
    
    y_train_pred, y_test_pred = [ y[0] for y in model.predict(x_train).tolist() ], [ y[0] for y in model.predict(x_test).tolist() ]
    
    r2, sdec = r2_score(y_train, y_train_pred), math.sqrt(mean_squared_error(y_train, y_train_pred))
    
    q2, sdep = leave_one_out(X=np.array(x_train), Y=np.array(y_train), M=model)
    
    scores = {
        'R2': r2,
        'Q2': q2,
        'SDEC': sdec,
        'SDEP': sdep,
    }
    
    return y_train_pred, y_test_pred, scores
    
