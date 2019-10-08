
import numpy as np
from sklearn.preprocessing import scale
from core import settings

def load_datasets(training, test, y_name):
    
    def print_data(X1, X2, Y1, Y2, O1, O2, y_name):
        classes, occurrences = np.unique(Y1, return_counts=True)[0].tolist(), np.unique(Y1, return_counts=True)[1].tolist()
        output = '\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1\n' % (len(O1), len(X1[0])) + 'Y (%s) = ' % (y_name) + ' + '.join(['%s (class %s)' % (occurrences[x], classes[x]) for x in range(len(classes))])
        classes, occurrences = np.unique(Y2, return_counts=True)[0].tolist(), np.unique(Y2, return_counts=True)[1].tolist()
        output += '\n\nTEST SET:\nN objects = %s\n' % (len(O2)) + 'Y (%s) = ' % (y_name) + ' + '.join(['%s (class %s)' % (occurrences[x], classes[x]) for x in range(len(classes))])
        print(output)
    
    def load_training_data(training, y_name):
        X,Y,O=[],[],[]
        l=0
        for line in open(training, 'r'):
            if l==0:
                V = str.split(line.strip(), ';')
                line = str.split(line.strip(), ';')[1:]
                Yind = line.index(y_name)
            else:
                O.append(line.split(';')[0])
                line = str.split(line.strip(), ';')[1:]
                X.append([float(line[x]) for x in range(len(line)) if x != Yind])
                try:
                    int(line[Yind])
                except:
                    Y.append(float(line[Yind]))
                else:
                    Y.append(int(line[Yind]))
            l+=1
        return X, Y, O, V
    
    def load_test_data(test, y_name):
        X,Y,O=[],[],[]
        l=0
        for line in open(test, 'r'):
            if l==0:
                line = str.split(line.strip(), ';')[1:]
                Yind = line.index(y_name)
            else:
                O.append(line.split(';')[0])
                line = str.split(line.strip(), ';')[1:]
                X.append([float(line[x]) for x in range(len(line)) if x != Yind])
                try:
                    int(line[Yind])
                except:
                    Y.append(float(line[Yind]))
                else:
                    Y.append(int(line[Yind]))
            l+=1
        return X, Y, O
    
    X1, Y1, O1, V = load_training_data(training, y_name)
    X2, Y2, O2 = load_test_data(test, y_name)
    
    X1, X2 = scale(X1).tolist(), scale(X2).tolist()
    
    if settings.VERBOSE==-1:
        print_data(X1, X2, Y1, Y2, O1, O2, y_name)
    
    return X1, X2, Y1, Y2, O1, O2, V
