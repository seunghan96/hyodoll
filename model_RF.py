import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,roc_auc_score


DATA_DIR = '/Users/LSH/Desktop/hyodoll/data'

def hypertuning_rscv(model, param_distn, n_iter,cv,X,y):
    rdmsearch = RandomizedSearchCV(model, param_distributions=param_distn, n_jobs=-1, n_iter=n_iter, cv=cv, random_state=19960729)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_params = {'max_depth':np.arange(6, 10), 
            'n_estimators':np.arange(30, 60),
            'min_samples_split':np.arange(4, 20)}
    
if __name__ == '__main__':
    data = pd.read_csv(os.path.join(DATA_DIR,'cluster_df_merged.csv'))
    dummy_col = pd.get_dummies(data['cluster'])
    dummy_col.columns = ['cluster_'+str(i) for i in dummy_col.columns]
    data = pd.concat([data,dummy_col],axis=1)

    N = data.shape[0]
    train_ratio = 0.8
    np.random.seed(19960729)
    train_index = list(np.random.choice(N, int(N*train_ratio), replace=False))
    test_index = list(set(range(N)) - set(train_index))
    
    X_cols = [x for x in data.columns if 'FA_' in x] + [x for x in data.columns if 'cluster_' in x]
    y_cols = [x for x in data.columns if 'emergency_month' in x] + [x for x in data.columns if 'emergency_time' in x]

    X = data[X_cols].values 
    X_train = X[train_index]
    X_test = X[test_index]

    thres_list = [0.5]
    predicted_df = pd.DataFrame(index=range(len(X_test)),columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    performance_index = ['f1','acc','auroc']
    performance_df = pd.DataFrame(index=performance_index,columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])

    

    for y_col in y_cols:
        y=data[y_col].values
        y_train = y[train_index]
        y_test = y[test_index]
        y_train[y_train>0]=1
        y_test[y_test>0]=1

        
        clf = RandomForestClassifier(random_state=19960729)
        #best_result = hypertuning_rscv(model=clf, param_distn=rf_params, n_iter=30,cv=5, X=X, y=y)
        #best_params = best_result[0]
        #clf.set_params(**best_params)
        clf.fit(X_train, y_train)
        #print('best_params : ', best_params)
        #----------------------------------------------#
        for thres in thres_list:
            y_pred = 1-clf.predict_proba(X_test)[:,0]
            y_pred_class = (y_pred>thres).astype('int')
            #----------------------------------------------#
            #score_cv = cross_val_score(clf, X, y, cv=5).mean()
            score_f1 = f1_score(y_test, y_pred_class)
            score_acc = np.mean((y_test==y_pred_class).astype('int'))
            score_auroc = roc_auc_score(y_test,y_pred)
            #----------------------------------------------#
            df_col='thres'+str(thres)+'_'+str(y_col)
            predicted_df[df_col] = y_pred
            performance_df[df_col] = [score_f1,score_acc,score_auroc]
            
    best_cols=[]
    for y_col in y_cols:
        tmp = performance_df[performance_df.columns[performance_df.columns.str.endswith(y_col)]].idxmax(axis=1)['f1']
        best_cols.append(tmp)

    print(performance_df[best_cols].mean(axis=1))