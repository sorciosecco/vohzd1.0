
import numpy as np

param_grids = {
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
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    'RF': {
        'n_estimators': np.arange(10,505,5).tolist(),
        'criterion_rf': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None] + np.arange(0.05,1.0,0.05).tolist(),
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
     'ETC': {
        'n_estimators': np.arange(10,505,5).tolist(),
        'criterion_rf': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None] + np.arange(0.05,1.0,0.05).tolist(),
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'AB': {
        'n_estimators': np.arange(10, 505, 5).tolist(),
        'algorithm': ['SAMME.R', 'SAMME'],
    },
    # GRADIENT BOOSTING (GB)
    #
    # learning_rate : float, optional (default=0.1)
    #     learning rate shrinks the contribution of each tree by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # subsample : float, optional (default=1.0)
    #     The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in
    #     Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators.
    #     Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    #
    # init : BaseEstimator, None, optional (default=None)
    #     An estimator object that is used to compute the initial predictions. init has to provide fit and predict.
    #     If None it uses loss.init_estimator.
    #
    'GB': {
        'n_estimators': np.arange(10,505,5).tolist(),
        'max_features': ['sqrt', 'log2', None] + np.arange(0.05,1.0,0.05).tolist(),
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'loss': ['deviance', 'exponential'],
        'criterion_gb': ['friedman_mse', 'mse', 'mae'],
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
    # tol : float, optional, default 1e-4
    #   Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
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
