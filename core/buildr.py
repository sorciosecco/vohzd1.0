
from sklearn.metrics import r2_score

from core import settings
from core.load import load_datasets
from core.regression import get_pls_predictions, get_ml_predictions

def run_pls(X_train, X_test, y_train, y_test):
    if settings.LATENT!=None:
        y_train_pred, y_test_pred, Scores = get_pls_predictions(x_train=X_train, x_test=X_test, y_train=y_train, latent_variables=settings.LATENT)
        print("\nR2 (TRA) = %s\nR2 (TES) = %s\n" % (round(Scores['R2'],3), round(r2_score(y_test, y_test_pred),3)))
    else:
        print("\nLV\tR2\tQ2\tSDEC\tSDEP")
        stop, best_q2, best_lv = False, 0, 0
        for lv in range(1, 11):
            y_train_pred, y_test_pred, Scores = get_pls_predictions(x_train=X_train, x_test=X_test, y_train=y_train, latent_variables=lv)
            if Scores['Q2'] < best_q2:
                stop=True
            else:
                best_q2, best_lv = Scores['Q2'], lv
                print('\t'.join([str(lv)] + [ str(round(Scores[k],3)) for k in list(Scores.keys()) ]))
            if stop:
                break
    return y_train_pred, y_test_pred

def build_regression_model(args):
    settings.NPARA=args.npara
    settings.PROBACUTOFF=args.probacutoff
    #settings.SAVEMODEL=args.savemodel
    settings.SAVEPRED=args.savepred
    #settings.MICROSPEC=args.microspec
    #settings.GRIDSEARCH=args.gridsearch
    settings.LATENT=args.latent
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, y_name=settings.ACTIVITY)
    if settings.VERBOSE==1:
        print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)\n\nTEST SET:\nN objects = %s\n" % (len(O1), len(X1[0]), settings.ACTIVITY, len(O2)))
    
    if settings.MODEL=="PLS":
        y_train_pred, y_test_pred = run_pls(X_train=X1, X_test=X2, y_train=Y1, y_test=Y2)
    else:
        Y1_pred, Y2_pred, Y1_prob, Y2_prob = modelling(X_train=X1, X_test=X2, Y_train=Y1, model_type=settings.MODEL, nondef_params=settings.NPARA, sm=settings.SAVEMODEL, mc=settings.MULTICLASS)
