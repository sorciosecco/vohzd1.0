
import pandas as pd

from core import settings
from core.load import load_datasets
from core.models import modelling
from core.metrics import calculate_class_scores, calculate_micro_scores
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

def make_micro_models(X1, X2, Y1, Y2, O1, O2, V):
    from sklearn.preprocessing import scale
    def define_micro_sets(D1, D2, variables):
        df1, df2 = pd.DataFrame([ [key]+D1[key] for key in sorted(list(D1.keys())) ], columns=variables), pd.DataFrame([ [key]+D2[key] for key in sorted(list(D2.keys())) ], columns=variables)
        df1_first, df2_first = df1.groupby('Objects').first(), df2.groupby('Objects').first()
        df1_o, df2_o = df1_first['Name'].values.tolist(), df2_first['Name'].values.tolist()
        df1_y, df2_y = df1_first[settings.ACTIVITY].values.tolist(), df2_first[settings.ACTIVITY].values.tolist()
        df1_x, df2_x = df1_first.loc[df1_first.index.values.tolist(), variables[2:-1]].values.tolist(), df2_first.loc[df2_first.index.values.tolist(), variables[2:-1]].values.tolist()
        D1_redu, D2_redu = { df1_o[i]: df1_x[i]+[df1_y[i]] for i in range(len(df1_o)) }, { df2_o[i]: df2_x[i]+[df2_y[i]] for i in range(len(df2_o)) }
        return D1_redu, D2_redu
    
    # S1 and S2 are dictionaries, for training and test respectively, of objects that HAVE NOT microspicies
    S1, S2 = { O1[i]: X1[i]+[Y1[i]] for i in range(len(O1)) if "_" not in O1[i] }, { O2[i]: X2[i]+[Y2[i]] for i in range(len(O2)) if "_" not in O2[i] }
    
    # M1 and M2 are dictionaries of objects that HAVE microspicies
    M1, M2 = { O1[i]: [O1[i].split('_')[0]]+X1[i]+[Y1[i]] for i in range(len(O1)) if "_" in O1[i] }, { O2[i]: [O2[i].split('_')[0]]+X2[i]+[Y2[i]] for i in range(len(O2)) if "_" in O2[i] }
    #print(len(S1), len(S2), len(M1), len(M2))
    
    i=0
    while i != 10:
        
        # returns the first microspecie for both sets
        R1, R2 = define_micro_sets(D1=M1, D2=M2, variables=['Name']+V)
        
        OXY1, OXY2 = S1.copy(), S2.copy()
        
        OXY1.update(R1)
        OXY2.update(R2)
        
        X1, X2, Y1, Y2, O1, O2 = [ OXY1[k][:-1] for k in sorted(list(OXY1.keys())) ], [ OXY2[k][:-1] for k in sorted(list(OXY2.keys())) ], [ OXY1[k][-1] for k in sorted(list(OXY1.keys())) ], [ OXY2[k][-1] for k in sorted(list(OXY2.keys())) ], sorted(list(OXY1.keys())), sorted(list(OXY2.keys()))
        
        Y1_pred, Y2_pred, Y1_prob, Y2_prob = modelling(X_train=scale(X1), X_test=scale(X2), Y_train=Y1, model_type=settings.MODEL, nondef_params=settings.NPARA, sm=settings.SAVEMODEL, mc=settings.MULTICLASS, gs=settings.GRIDSEARCH)
        
        NoU, uncertain = calculate_micro_scores(Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, pc=settings.PROBACUTOFF)
        
        for u in uncertain:
            if u in list(M1.keys()):
                if len([1 for k in list(M1.keys()) if k.find(u.split("_")[0]) != -1])==1:
                    S1[u] = M1[u][1:]
                del M1[u]
            elif u in list(M2.keys()):
                if len([1 for k in list(M2.keys()) if k.find(u.split("_")[0]) != -1])==1:
                    S2[u] = M2[u][1:]
                del M2[u]
        #print(i, len(S1), len(S2), len(M1), len(M2))
        i+=1
    

def build_classification_model(args):
    settings.NPARA=args.npara
    settings.MULTICLASS=args.multiclass
    settings.PROBACUTOFF=args.probacutoff
    settings.SAVEMODEL=args.savemodel
    settings.SAVEPRED=args.savepred
    settings.MICROSPEC=args.microspec
    settings.GRIDSEARCH=args.gridsearch
    settings.VERBOSE=-1
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, y_name=settings.ACTIVITY)
    
    if settings.GRIDSEARCH:
        gridsearchcv(X=X1, Y=Y1, grid=param_grids[settings.MODEL])
    else:
        if settings.MICROSPEC==False:
            make_one_model(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2)
        else:
            make_micro_models(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2, V=V)
