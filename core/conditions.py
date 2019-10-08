
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
# ENSEMBLE METHODS BASED ON DECISION TREES
# Two averaging algorithms based on randomized decision trees: the RandomForest algorithm and the Extra-Trees method.
# Both algorithms are perturb-and-combine techniques specifically designed for trees.
# This means a diverse set of classifiers is created by introducing randomness in the classifier construction.
# The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
# ----------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST (RF)
# In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample)
# from the training set. In addition, when splitting a node during the construction of the tree, the chosen split
# is no longer the best split among all features. Instead, the split that is picked is the best split among a random
# subset of the features. As a result of this randomness, the bias of the forest usually slightly increases (with
# respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than
# compensating for the increase in bias, hence yielding an overall better model.
# ----------------------------------------------------------------------------------------------------------------------
# In contrast to the original publication, the scikit-learn implementation combines classifiers by averaging their
# probabilistic prediction, instead of letting each classifier vote for a single class.
# ----------------------------------------------------------------------------------------------------------------------
# EXTRA-TREE (ET)
# In extremely randomized trees, randomness goes one step further in the way splits are computed. As in random forests,
# a random subset of candidate features is used, but instead of looking for the most discriminative thresholds,
# thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is
# picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense
# of a slightly greater increase in bias.
# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS (see details for the methods below)
# The main parameters to adjust when using these methods are n_estimators and max_features.
# The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute.
# In addition, note that results will stop getting significantly better beyond a critical number of trees.
# The latter is the size of the random subsets of features to consider when splitting a node.
# The lower the greater the reduction of variance, but also the greater the increase in bias.
# Empirical good default values are max_features=n_features for regression problems, and max_features=sqrt(n_features)
# for classification tasks (where n_features is the number of features in the data).
# Good results are often achieved when setting max_depth=None in combination with min_samples_split=1
# (i.e., when fully developing the trees). Bear in mind though that these values are usually not optimal,
# and might result in models that consume a lot of ram. The best parameter values should always be cross-validated.
# In addition, note that in random forests, bootstrap samples are used by default (bootstrap=True) while the default
# strategy for extra-trees is to use the whole dataset (bootstrap=False).
# When using bootstrap sampling the generalization error can be estimated on the left out or out-of-bag samples.
# This can be enabled by setting oob_score=True.
# ----------------------------------------------------------------------------------------------------------------------

param_grids = {
    # RANDOM FOREST (RF)
    # n_estimators : integer, optional (default=100)
    #     The number of trees in the forest.
    #
    # criterion : string, optional (default="gini")
    #     The function to measure the quality of a split. Supported criteria are
    #     "gini" for the Gini impurity and "entropy" for the information gain.
    #     Note: this parameter is tree-specific.
    #
    # max_features : int, float, string or None, optional (default="auto")
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered
    #         at each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features) (same as "auto").
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #     Note: the search for a split does not stop until at least one valid partition
    #     of the node samples is found, even if it requires to effectively inspect
    #     more than max_features features.
    #     Note: this parameter is tree-specific.
    #
    # max_depth : integer or None, optional (default=None)
    #     The maximum depth of the tree. If None, then nodes are expanded until all
    #     leaves are pure or until all leaves contain less than min_samples_split
    #     samples. Ignored if max_leaf_nodes is not None.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples in newly created leaves.
    #     A split is discarded if after the split, one of the leaves would contain
    #     less then min_samples_leaf samples.
    #     Note: this parameter is tree-specific.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a
    #     leaf node.
    #     Note: this parameter is tree-specific.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion.
    #     Best nodes are defined as relative reduction in impurity. If None then
    #     unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #     Note: this parameter is tree-specific.
    #
    # bootstrap : boolean, optional (default=True)
    #     Whether bootstrap samples are used when building trees.
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    # n_jobs : integer, optional (default=1)
    #     The number of jobs to run in parallel for both fit and predict.
    #     If -1, then the number of jobs is set to the number of cores.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator;
    #     If RandomState instance, random_state is the random number generator;
    #     If None, the random number generator is the RandomState instance used by np.random.
    #
    # verbose : int, optional (default=0)
    #     Controls the verbosity of the tree building process.
    #
    # class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
    #     Weights associated with classes in the form {class_label: weight}.
    #     If not given, all classes are supposed to have weight one.
    #     For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    #     The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class
    #     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    #     The "balanced_subsample" mode is the same as "balanced" except that weights are computed based on the
    #     bootstrap sample for every tree grown.
    #     For multi-output, the weights of each column of y will be multiplied.
    #     Note that these weights will be multiplied with sample_weight
    #     (passed through the fit method) if sample_weight is specified.
    'RF': {
        'n_estimators': np.arange(10,210,10).tolist(),
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        
        #'min_samples_split': [2],
        #'min_samples_leaf': [1],
        #'min_weight_fraction_leaf': [0.0],
        #'bootstrap': [True],
        #'oob_score': [False],
    },
    # EXTRA-TREE (ET)
    #     n_estimators : integer, optional (default=100)
    #     The number of trees in the forest.
    #
    # criterion : string, optional (default="gini")
    #     The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and
    #     "entropy" for the information gain. Note: this parameter is tree-specific.
    #
    # max_features : int, float, string or None, optional (default="auto")
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered at
    #         each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features).
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #     Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    #     even if it requires to effectively inspect more than max_features features.
    #     Note: this parameter is tree-specific.
    #
    # max_depth : integer or None, optional (default=None)
    #     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
    #     contain less than min_samples_split samples. Ignored if max_leaf_nodes is not None.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples in newly created leaves. A split is discarded if after the split, one of the
    #     leaves would contain less then min_samples_leaf samples. Note: this parameter is tree-specific.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a leaf node.
    #     Note: this parameter is tree-specific.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    #     If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #     Note: this parameter is tree-specific.
    #
    # bootstrap : boolean, optional (default=False)
    #     Whether bootstrap samples are used when building trees.
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    # n_jobs : integer, optional (default=1)
    #     The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the
    #     number of cores.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
    #     the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #
    # class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
    #     Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to
    #     have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    #     The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class
    #     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    #     The "balanced_subsample" mode is the same as "balanced" except that weights are computed based on the
    #     bootstrap sample for every tree grown.
    #     For multi-output, the weights of each column of y will be multiplied.
    #     Note that these weights will be multiplied with sample_weight (passed through the fit method) if
    #     sample_weight is specified.
     'ETC': {
        'n_estimators': np.arange(10,210,10).tolist(),
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        #'bootstrap': [False, True],
        
        #'min_samples_split': [2],
        #'min_samples_leaf': [1],
        #'min_weight_fraction_leaf': [0.0],
        #'oob_score': [False],
        #'n_jobs': [1],
        #'random_state': [999],
    },
    # ADA BOOSTING (AB)
    # base_estimator : object, optional (default=DecisionTreeClassifier)
    #     The base estimator from which the boosted ensemble is built. Support for sample weighting is required,
    #     as well as proper classes_ and n_classes_ attributes.
    #
    # n_estimators : integer, optional (default=50)
    #     The maximum number of estimators at which boosting is terminated. In case of perfect fit,
    #     the learning procedure is stopped early.
    #
    # learning_rate : float, optional (default=1.)
    #     Learning rate shrinks the contribution of each classifier by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # algorithm : {"SAMME", "SAMME.R"}, optional (default="SAMME.R")
    #     If SAMME.R then use the SAMME.R real boosting algorithm. base_estimator must support calculation of
    #     class probabilities. If SAMME then use the SAMME discrete boosting algorithm. The SAMME.R algorithm
    #     typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
    #     is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    'AB': {
        'n_estimators': np.arange(10, 510, 10).tolist(),
        'algorithm': ['SAMME.R', 'SAMME'],
        
        #'base_estimator': [tree.DecisionTreeClassifier()],
        #'learning_rate': [1.0],
        #'random_state': [999],
    },
    # GRADIENT BOOSTING (GB)
    # loss : {"deviance", "exponential"}, optional (default="deviance")
    #     loss function to be optimized. "deviance" refers to deviance (= logistic regression) for classification
    #     with probabilistic outputs. For loss "exponential" gradient boosting recovers the AdaBoost algorithm.
    #
    # learning_rate : float, optional (default=0.1)
    #     learning rate shrinks the contribution of each tree by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # n_estimators : int (default=100)
    #     The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number
    #     usually results in better performance.
    #
    # max_depth : integer, optional (default=3)
    #     maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.
    #     Tune this parameter for best performance; the best value depends on the interaction of the input variables.
    #     Ignored if max_leaf_nodes is not None.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples required to be at a leaf node.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a leaf node.
    #
    # subsample : float, optional (default=1.0)
    #     The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in
    #     Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators.
    #     Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    #
    # max_features : int, float, string or None, optional (default=None)
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered at
    #         each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features).
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #
    #     Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
    #     Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    #     even if it requires to effectively inspect more than max_features features.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    #     If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #
    # init : BaseEstimator, None, optional (default=None)
    #     An estimator object that is used to compute the initial predictions. init has to provide fit and predict.
    #     If None it uses loss.init_estimator.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
    #     is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #
    'GB': {
        'n_estimators': np.arange(10,210,10).tolist(),
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_features': ['auto', 'log2', None],
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        
        #'loss': ['deviance'],
        #'learning_rate': [0.1],
        #'subsample': [1.0],
        #'min_samples_split': [2],
        #'min_samples_leaf': [1],
        #'min_weight_fraction_leaf': [0.0],
        #'init': [None],
        #'random_state': [999],
    },
    'LDA': {
        'shrinkage': np.arange(0.01,1.01,0.01).tolist() + [None, 'auto'],
        'solver': ['svd', 'lsqr', 'eigen'],
    },
    'SVM': {
        'C': np.arange(1,11,1).tolist(),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale'] + np.arange(0.001,0.016,0.001).tolist(),
        'degree': [1,2,3],
        'class_weight': [None, 'balanced'],
    },
    # MULTI-LAYER PERCEPTION (MLP)
    # hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
    #      It is a tuple of layers dimensions (default is 1 layer of 100 neurons).
    #
    # activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
    #       Activation function for the hidden layer:
    #           ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
    #           ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
    #           ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
    #           ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
    #
    # solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
    #   The solver for weight optimization:
    #       ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
    #       ‘sgd’ refers to stochastic gradient descent.
    #       ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
    #   Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.
    #
    # alpha : float, optional, default 0.0001
    #   L2 penalty (regularization term) parameter.
    #
    # batch_size : int, optional, default ‘auto’
    #   Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
    #
    # learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
    #   Learning rate schedule for weight updates:
    #       ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
    #       ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
    #       ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if   ‘early_stopping’ is on, the current learning rate is divided by 5.
    #   Only used when solver='sgd'.
    #
    # learning_rate_init : double, optional, default 0.001
    #   The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.
    #
    # power_t : double, optional, default 0.5
    #   The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.
    #
    # max_iter : int, optional, default 200
    #   Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
    #
    # shuffle : bool, optional, default True
    #   Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.
    #
    # random_state : int, RandomState instance or None, optional, default None
    #   If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #
    # tol : float, optional, default 1e-4
    #   Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    #
    # verbose : bool, optional, default False
    #   Whether to print progress messages to stdout.
    #
    # warm_start : bool, optional, default False
    #   When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. See the Glossary.
    #
    # momentum : float, default 0.9
    #   Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.
    #
    # nesterovs_momentum : boolean, default True
    #   Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.
    #
    #   early_stopping : bool, default False
    #   Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’
    #
    #   validation_fraction : float, optional, default 0.1
    #   The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True
    #
    #   beta_1 : float, optional, default 0.9
    #   Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’
    #
    #   beta_2 : float, optional, default 0.999
    #   Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’
    #
    #   epsilon : float, optional, default 1e-8
    #   Value for numerical stability in adam. Only used when solver=’adam’
    #
    #   n_iter_no_change : int, optional, default 10
    #   Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’
    #
    #'MLP': {
        #'hidden_layer_sizes': np.arange(1,11,1).tolist(),
        #'max_iter': np.arange(1,11,1).tolist(),
        #'max_iter': ['linear', 'poly', 'rbf', 'sigmoid'],
    #},
}
