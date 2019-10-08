
#NB: to implement for multiclass with single score!!!!!

import itertools
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, recall_score, confusion_matrix

from core import settings
from core import parameters
from core.models import define_model_for_optimization

m=mp.Manager()
q=m.Queue()

def cross_validation(x, y, cv, model):
    X, Y = np.array(x), np.array(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=settings.SEED)
    scores, SE, SP, MCC = [], [], [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test).tolist()
        if settings.MULTICLASS:
            s = sum([1 for x in range(len(y_test)) if y_pred[x]!=y_test[x]]) / len(y_test)
            scores.append(s)
        else:
            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
            se, sp = recall_score(y_test, y_pred), TN/(TN + FP)
            mcc = matthews_corrcoef(y_test, y_pred)
            SE.append(se)
            SP.append(sp)
            MCC.append(mcc)
    if settings.MULTICLASS:
        return [np.mean(scores)]
    else:
        return [np.mean(SE), np.mean(SP), np.mean(MCC)]

def compute_model(model):
    X, Y, m = model[0], model[1], model[-1]
    try:
        m.fit(X, Y)
    except:
        scores = [-99,-99,-99]
    else:
        scores = cross_validation(x=X, y=Y, cv=5, model=m)
    
    q.put(1)
    size=q.qsize()
    message=""
    if settings.VERBOSE==0:
        message += '['+str(round(size*100/settings.N,2))+' %] of models completed'
    elif settings.VERBOSE==1:
        message += '[%s %] of models completed\n\tSE = %s\n\tSP = %s\n\tMCC = %s' % (round(size*100/settings.N,2), round(scores[0],3), round(scores[1],3), round(scores[-1],3))
    print(message)
    
    if settings.MULTICLASS:
        params = { k.split('__')[-1]: m.get_params()[k] for k in list(m.get_params().keys()) }
    else:
        params = m.get_params()
    
    return params, scores

def gridsearchcv(X, Y, grid):
    if settings.CPUS == None:
        ncpus=mp.cpu_count()
    else:
        ncpus=args.cpus
    
    names, values = list(grid.keys()), list(itertools.product(*grid.values()))
    settings.N=len(values)
    
    models=[]
    for v in values:
        combo={names[i]: v[i] for i in range(len(v))}
        for p in list(combo.keys()):
            if p=='n_estimators':
                parameters.n_estimators = combo[p]
            elif p=='criterion':
                parameters.criterion = combo[p]
            elif p=='max_features':
                parameters.max_features = combo[p]
            elif p=='max_depth':
                parameters.max_depth = combo[p]
            elif p=='max_leaf_nodes':
                parameters.max_leaf_nodes = combo[p]
            elif p=='class_weight':
                parameters.class_weight = combo[p]
            elif p=='bootstrap':
                parameters.bootstrap = combo[p]
            elif p=='algorithm':
                parameters.algorithm = combo[p]
            elif p=='shrinkage':
                parameters.shrinkage = combo[p]
            elif p=='solver':
                parameters.solver = combo[p]
            elif p=='C':
                parameters.C = combo[p]
            elif p=='kernel':
                parameters.kernel = combo[p]
            elif p=='gamma':
                parameters.gamma = combo[p]
            elif p=='degree':
                parameters.degree = combo[p]
        model = define_model_for_optimization(mt=settings.MODEL, ndp=True, mc=settings.MULTICLASS)
        models.append((X, Y, model))
    
    pool=mp.Pool(ncpus)
    results = pool.map_async(compute_model, models)
    pool.close()
    pool.join()
    #results=[]
    #for i, m in enumerate(models):
        #r1, r2 = compute_model(m)
        #results.append([r1,r2])
    
    outfile=open("opt_results_%s.csv" % settings.MODEL, "w")
    if settings.MULTICLASS: outfile.write(';'.join(['model_id'] + sorted(names) + ['EE\n']))
    else: outfile.write(';'.join(['model_id'] + sorted(names) + ['SE','SP','MCC\n']))
    #for i, r in enumerate(results):
    for i, r in enumerate(results.get()):
        outfile.write(';'.join([str(i)] + [str(r[0][k]) for k in sorted(list(r[0].keys())) if k in names] + [str(s) for s in r[-1]]) + '\n')
    outfile.close()
