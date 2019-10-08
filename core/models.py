
import os
import pickle

from core import settings
from core import parameters

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier

def algorithm_setup(model_type, nondef_params):
    
    # Ada Boost (AB)
    if model_type=="AB":
        from sklearn.tree import DecisionTreeClassifier
        if nondef_params:
            model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=parameters.n_estimators, algorithm=parameters.algorithm, random_state=settings.SEED)
        else:
            model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)
                                     , n_estimators=100
                                     , algorithm='SAMME.R'
                                     , learning_rate=1.0
                                     , random_state=settings.SEED)
    
    # Extra Trees Classifier (ETC)
    if model_type=="ETC":
        if nondef_params:
            model=ExtraTreesClassifier(n_estimators=parameters.n_estimators, class_weight=parameters.class_weight, criterion=parameters.criterion, max_depth=parameters.max_depth, max_features=parameters.max_features, max_leaf_nodes=parameters.max_leaf_nodes, random_state=settings.SEED)
        else:
            model=ExtraTreesClassifier(random_state=settings.SEED
                                       , n_estimators=100
                                       , bootstrap=True
                                       , class_weight=None
                                       , criterion='gini'
                                       , max_depth=None
                                       , max_features='auto'
                                       , max_leaf_nodes=None
                                       , min_samples_split=2
                                       , min_samples_leaf=1
                                       , min_weight_fraction_leaf=0.0
                                       , min_impurity_decrease=0.0
                                       , min_impurity_split=None
                                       , oob_score=False
                                       , n_jobs=-1
                                       , warm_start=False
                                       )
    
    # Gradient Boosting (GB)
    if model_type=="GB":
        if nondef_params:
            model=GradientBoostingClassifier(n_estimators=parameters.n_estimators, max_depth=parameters.max_depth, max_features=parameters.max_features, max_leaf_nodes=parameters.max_leaf_nodes, random_state=settings.SEED)
        else:
            model=GradientBoostingClassifier(random_state=settings.SEED
                                             , n_estimators=100
                                             , max_depth=3
                                             , max_features=None
                                             , max_leaf_nodes=None
                                             , learning_rate=0.1
                                             , loss='deviance'
                                             , subsample=1.0
                                             , criterion='friedman_mse'
                                             , min_samples_split=2
                                             , min_samples_leaf=1
                                             , min_weight_fraction_leaf=0.0
                                             , min_impurity_decrease=0.0
                                             , min_impurity_split=None
                                             , init=None
                                             , warm_start=False
                                             , presort='auto'
                                             , validation_fraction=0.1
                                             , n_iter_no_change=None
                                             , tol=0.0001
                                             )
    
    # k-Nearest Neighbors (kNN)
    if model_type=="kNN":
        model=KNeighborsClassifier(n_neighbors=5
                                   , weights='uniform'
                                   , algorithm='auto'
                                   , leaf_size=30
                                   , p=2
                                   , metric='minkowski'
                                   , metric_params=None
                                   , n_jobs=1)
    
    # Linear Discriminant Analysis (LDA)
    if model_type=="LDA":
        if nondef_params:
            model=LinearDiscriminantAnalysis(solver=parameters.solver, shrinkage=parameters.shrinkage)
        else:
            model=LinearDiscriminantAnalysis(solver='svd'
                                             , shrinkage=None
                                             , priors=None
                                             , n_components=None
                                             , store_covariance=False
                                             , tol=0.0001)
    
    # Multi-layer Perception (MLP)
    if model_type=="MLP":
        if nondef_params:
            model=MLPClassifier(random_state=settings.SEED, hidden_layer_sizes=parameters.hidden_layer_sizes, max_iter=parameters.max_iter)
        else:
            model=MLPClassifier(random_state=settings.SEED
                                , hidden_layer_sizes=(100,)
                                , activation='relu'
                                , solver='adam'
                                , alpha=0.0001
                                , batch_size='auto'
                                , learning_rate='constant'
                                , learning_rate_init=0.001
                                , power_t=0.5
                                , max_iter=500
                                , shuffle=True
                                , tol=0.0001
                                , warm_start=False
                                , momentum=0.9
                                , nesterovs_momentum=True
                                , early_stopping=False
                                , validation_fraction=0.1
                                , beta_1=0.9
                                , beta_2=0.999
                                , epsilon=1e-08
                                , n_iter_no_change=10
                                )
    
    # Random Forest (RF)
    if model_type=="RF":
        if nondef_params:
            model=RandomForestClassifier(n_estimators=parameters.n_estimators, class_weight=parameters.class_weight, criterion=parameters.criterion, max_depth=parameters.max_depth, max_features=parameters.max_features, max_leaf_nodes=parameters.max_leaf_nodes, random_state=settings.SEED)
        else:
            model=RandomForestClassifier(random_state=settings.SEED
                                         , n_estimators=100
                                         , class_weight=None
                                         , criterion='gini'
                                         , max_depth=None
                                         , max_features='auto'
                                         , max_leaf_nodes=None
                                         , min_samples_split=2
                                         , min_samples_leaf=1
                                         , min_weight_fraction_leaf=0.0
                                         , min_impurity_decrease=0.0
                                         , min_impurity_split=None
                                         , bootstrap=True
                                         , oob_score=False
                                         , n_jobs=-1
                                         , warm_start=False
                                         )
    
    # Support Vector Machines (SVM)
    if model_type=="SVM":
        if nondef_params:
            model=SVC(probability=True, C=parameters.C, degree=parameters.degree, gamma=parameters.gamma, kernel=parameters.kernel, random_state=settings.SEED, class_weight=parameters.class_weight)
        else:
            model=SVC(random_state=settings.SEED
                , C=1.0
                , kernel='rbf'
                , degree=3
                , gamma='auto'
                , coef0=0.0
                , shrinking=True
                , probability=True
                , tol=0.001
                , cache_size=200
                , class_weight=None
                , max_iter=-1
                , decision_function_shape='ovr'
                )
    
    return model

def save_model(model, model_type):
    model_filename = os.path.join(os.getcwd(), model_type+'.model')
    picklefile = open(model_filename, 'wb')
    pickle.dump(model, picklefile)
    picklefile.close()

def define_model_for_optimization(mt, ndp, mc):
    model=algorithm_setup(model_type=mt, nondef_params=ndp)
    if mc:
        model=OneVsRestClassifier(model, n_jobs=1)
    return model

def modelling(X_train, X_test, Y_train, model_type, nondef_params, sm, mc):
    if settings.VERBOSE:
        print('\nModelling in progress...')
    
    model=algorithm_setup(model_type, nondef_params)
    
    if mc:
        model=OneVsRestClassifier(model, n_jobs=1)
        #model=OneVsOneClassifier(model, n_jobs=1)
    
    model.fit(X_train, Y_train)
    
    Y1_pred, Y2_pred, Y1_prob, Y2_prob = model.predict(X_train).tolist(), model.predict(X_test).tolist(), model.predict_proba(X_train).tolist(), model.predict_proba(X_test).tolist()
    
    if sm:
        save_model(model, model_type)
    
    return Y1_pred, Y2_pred, Y1_prob, Y2_prob
