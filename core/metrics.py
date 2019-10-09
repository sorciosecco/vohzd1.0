
from core import settings

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

def calculate_pls_metrics(y_train_exp, y_train_pred, y_test_exp, y_test_pred, lco, hco):
    
    def get_classifications(ye, yp, l, h):
        R, t_list = [], []
        if l!=None and h!=None:
            thresholds = [ [l,h] ]
        else:
            thresholds = [ [x,y] for x in np.arange(0, 1.01, 0.01).tolist() for y in np.arange(0, 1.01, 0.01).tolist() ]
        
        for t in thresholds:
            if sorted(t) not in t_list:
                t_list.append(t)
                Yp, Ye = [], []
                UNC=0
                for y in range(len(yp)):
                    if yp[y] > t[1]:
                        Ye.append(ye[y])
                        Yp.append(1)
                    elif yp[y] < t[0]:
                        Ye.append(ye[y])
                        Yp.append(0)
                    else:
                        UNC+=1
                TN, FP, FN, TP = confusion_matrix(Ye, Yp).ravel()
                ACC, F1, MCC = round(accuracy_score(Ye, Yp),2), round(f1_score(Ye, Yp),2), round(matthews_corrcoef(Ye, Yp),2)
                pPREC, nPREC = round(precision_score(Ye, Yp),2), round(TN/(TN+FN),2)
                SE, SP = round(recall_score(Ye, Yp),2), round(TN/(TN + FP),2)
                COV = round((len(ye)-UNC)*100/len(ye),1)
                R.append([t[0],t[1],TP,FN,TN,FP,UNC,ACC,SE,SP,pPREC,nPREC,F1,MCC,COV])
        
        return R
    
    if lco!=None and hco!=None:
        print('\nSet\tL\tH\tTP\tFN\tTN\tFP\tUNC\tACC\tSE\tSP\tPREC+\tPREC-\tF1\tMCC\tTotal coverage')
    
    for i, Set in enumerate([(y_train_exp, y_train_pred), (y_test_exp, y_test_pred)]):
        
        results = get_classifications(ye=Set[0], yp=Set[1], l=lco, h=hco)
        
        if lco!=None and hco!=None:
            if i==0:
                print('TRAIN\t'+'\t'.join([ str(r) for r in results[0] ]))
            else:
                print('TEST\t'+'\t'.join([ str(r) for r in results[0] ]))
        else:
            if i==0:
                ocsv=open('scores_PLS_training.csv', 'w')
            else:
                ocsv=open('scores_PLS_test.csv', 'w')
            ocsv.write('L;H;TP;FN;TN;FP;UNC;ACC;SE;SP;PREC+;PREC-;F1;MCC;Total coverage\n')
            for r in results:
                ocsv.write(';'.join([ str(l) for l in r ])+'\n')
            ocsv.close()

def calculate_class_scores(Y1_exp, Y1_pred, Y1_prob, Y2_exp, Y2_pred, Y2_prob, O1, O2, mc, pc):
    
    def get_uncertain(Y, threshold):
        Ye, Yp, Yu = [], [], []
        for y in range(len(Y[0])):
            if max(Y[2][y]) > threshold:
                Ye.append(Y[0][y])
                Yp.append(Y[1][y])
            else:
                Yu.append(Y[0][y])
        return Ye, Yp, Yu
    
    if mc:
        scores=["Set\tEE\tEEw"]
        if pc==None: t=0.0
        else: t=pc
        
        for i, Y in enumerate([(Y1_exp, Y1_pred, Y1_prob, O1), (Y2_exp, Y2_pred, Y2_prob, O2)]):
            classes, occurrences = np.unique(Y[0], return_counts=True)[0].tolist(), np.unique(Y[0], return_counts=True)[1].tolist()
            
            Ye, Yp, Yu = get_uncertain(Y, threshold=t)
            
            EE = sum([1 for x in range(len(Ye)) if Yp[x]!=Ye[x]]) / len(Ye)
            if i==1:
            #if i!=-1:
                message="\nCorrect test predictions -> "+str(round(100-EE*100))+"%\n\nBDDCS\tPRED_1\tPRED_2\tPRED_3\tPRED_4\tUNC\tCORR_%"
                for c, bddcs_exp in enumerate(classes):
                    NoC = sum([1 for y in range(len(Ye)) if Ye[y]==bddcs_exp and Ye[y]==Yp[y]])
                    C_perc = round(NoC*100/occurrences[c])
                    message += '\n' + '\t'.join([str(bddcs_exp)] + [str(len([1 for y in range(len(Ye)) if Ye[y]==bddcs_exp and Yp[y]==bddcs_pred])) for bddcs_pred in classes] + [str(len([1 for y in range(len(Yu)) if Yu[y]==bddcs_exp]))] + [str(C_perc)])
                print(message)
    else:
        scores=["Set\tTP\tFN\tTN\tFP\tUNC\tACC\tSE\tSP\tPREC+\tPREC-\tF1\tMCC\tCoverage"]
        if pc==None: t=0.5
        else: t=pc
        
        for i, Y in enumerate([(Y1_exp, Y1_pred, Y1_prob), (Y2_exp, Y2_pred, Y2_prob)]):
            Ye, Yp, Yu = get_uncertain(Y, threshold=t)
            TN, FP, FN, TP = confusion_matrix(Ye, Yp).ravel()
            ACC = round(accuracy_score(Ye, Yp),2)
            F1 = round(f1_score(Ye, Yp),2)
            MCC = round(matthews_corrcoef(Ye, Yp),2)
            SE, SP = round(recall_score(Ye, Yp),2), round(TN/(TN+FP),2)
            pPREC, nPREC = round(precision_score(Ye, Yp),2), round(TN/(TN+FN),2)
            COV = round((len(Y[0])-len(Yu))*100/len(Y[0]))
            if i==0:
                scores.append("\t".join([str(s) for s in ["Train",TP,FN,TN,FP,len(Yu),ACC,SE,SP,pPREC,nPREC,F1,MCC,COV]]))
            else:
                scores.append("\t".join([str(s) for s in ["Test",TP,FN,TN,FP,len(Yu),ACC,SE,SP,pPREC,nPREC,F1,MCC,COV]]))
        print("\n"+"\n".join(scores))
    return "\n".join(scores)
