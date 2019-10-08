##### (RF ETC AB GB) parameters
n_estimators=300

## RF ETC GB
#max_features='sqrt'
#max_features='log2'
max_features=None

max_depth=None
max_leaf_nodes=None

## RF ETC SVM
class_weight=None
#class_weight='balanced'
#class_weight='balanced_subsample'

## RF ETC
#criterion='gini'
criterion='entropy'

## AB
algorithm='SAMME.R'

## ETC
#bootstrap=True

## LDA MLP
solver='lsqr'

## LDA
shrinkage=0.95

## SVM
C=9
degree=1
gamma=0.005
kernel='rbf'

## MLP
max_iter=500
#hidden_layer_sizes=(1000)
hidden_layer_sizes=(200,50)
#hidden_layer_sizes=(200,150,100)
#hidden_layer_sizes=(200,150,100,50)

#activation='logistic'
#alpha=0.0001
#batch_size='auto'
#learning_rate='constant'
#learning_rate_init=0.001
#power_t=0.5
#shuffle=True
#tol=0.0001
#warm_start=False
#momentum=0.9
#nesterovs_momentum=True
#early_stopping=False
#validation_fraction=0.1
#beta_1=0.9
#beta_2=0.999
#epsilon=1e-08
#n_iter_no_change=10
