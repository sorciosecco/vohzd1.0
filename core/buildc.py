
import pandas as pd

from core import settings
from core.load import load_datasets
from core.models import modelling
from core.metrics import calculate_class_scores
from core.save import save_predictions, save_multi_predictions
from core.conditions import param_grids
from core.optimization import gridsearchcv

def make_one_model(X1, X2, Y1, Y2, O1, O2):
    Y1_pred, Y2_pred, Y1_prob, Y2_prob = modelling(X_train=X1, X_test=X2, Y_train=Y1, model_type=settings.MODEL, nondef_params=settings.NPARA, sm=settings.SAVEMODEL, mc=settings.MULTICLASS)
    
    scores = calculate_class_scores(Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, mc=settings.MULTICLASS, pc=settings.PROBACUTOFF)
    
    if settings.SAVEPRED:
        if settings.MULTICLASS:
            save_multi_predictions(name=settings.MODEL, Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2)
        else:
            save_predictions(name=settings.MODEL, Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, pc=settings.PROBACUTOFF)
        

def build_classification_model(args):
    settings.NPARA=args.npara
    settings.MULTICLASS=args.multiclass
    settings.PROBACUTOFF=args.probacutoff
    settings.SAVEMODEL=args.savemodel
    settings.SAVEPRED=args.savepred
    settings.GRIDSEARCH=args.gridsearch
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, y_name=settings.ACTIVITY)
    
    if settings.GRIDSEARCH:
        gridsearchcv(X=X1, Y=Y1, grid=param_grids[settings.MODEL])
    else:
        make_one_model(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2)
    
