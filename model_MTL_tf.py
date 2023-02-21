import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import f1_score,roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

DATA_DIR = '/Users/seunghan96/Desktop/hyodoll_yonsei/data/'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default = 200, help = '# of epochs')

args = parser.parse_args()

epochs = args.epoch

if __name__ == '__main__':
    data = pd.read_csv(DATA_DIR+'cluster_df_merged.csv')
    dummy_col = pd.get_dummies(data['cluster'])
    dummy_col.columns = ['cluster_'+str(i) for i in dummy_col.columns]
    data = pd.concat([data,dummy_col],axis=1)

    X_cols = [x for x in data.columns if 'FA_' in x] + [x for x in data.columns if 'cluster_' in x]
    y_cols = [x for x in data.columns if 'emergency_month' in x] + [x for x in data.columns if 'emergency_time' in x]
    X = data[X_cols].values 
    
    thres_list= [0.2,0.4,0.6,0.8]
    predicted_df=pd.DataFrame(index=data.index,columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    performance_index=['f1','acc','auroc']
    performance_df=pd.DataFrame(index=performance_index,columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    
    X=data[X_cols].values 
    y=data[y_cols].values
    y[y>0]=1
    

    #------------------------------------#
    posts_input = Input(shape=(X.shape[1],), name='posts')
    #------------------------------------#
    x = layers.Dense(32,activation='relu')(posts_input)
    #------------------------------------#
    x_month = layers.Dense(64,activation='relu')(x)
    x_time = layers.Dense(64,activation='relu')(x)

    #------------------------------------#
    pred1=layers.Dense(8,activation='relu')(x_time)
    pred2=layers.Dense(8,activation='relu')(x_time)
    pred3=layers.Dense(8,activation='relu')(x_time)
    pred4=layers.Dense(8,activation='relu')(x_time)
    pred5=layers.Dense(8,activation='relu')(x_time)
    pred6=layers.Dense(8,activation='relu')(x_time)
    pred7=layers.Dense(8,activation='relu')(x_time)
    pred8=layers.Dense(8,activation='relu')(x_time)
    pred9=layers.Dense(8,activation='relu')(x_time)
    pred10=layers.Dense(8,activation='relu')(x_time)
    pred11=layers.Dense(8,activation='relu')(x_time)
    pred12=layers.Dense(8,activation='relu')(x_time)
    pred13=layers.Dense(8,activation='relu')(x_time)
    pred14=layers.Dense(8,activation='relu')(x_time)
    pred15=layers.Dense(8,activation='relu')(x_time)
    pred16=layers.Dense(8,activation='relu')(x_time)
    #-----------------------------------------------------------#
    pred1=layers.Dense(1,activation='sigmoid',name=y_cols[0])(pred1)
    pred2=layers.Dense(1,activation='sigmoid',name=y_cols[1])(pred2)
    pred3=layers.Dense(1,activation='sigmoid',name=y_cols[2])(pred3)
    pred4=layers.Dense(1,activation='sigmoid',name=y_cols[3])(pred4)
    pred5=layers.Dense(1,activation='sigmoid',name=y_cols[4])(pred5)
    pred6=layers.Dense(1,activation='sigmoid',name=y_cols[5])(pred6)
    pred7=layers.Dense(1,activation='sigmoid',name=y_cols[6])(pred7)
    pred8=layers.Dense(1,activation='sigmoid',name=y_cols[7])(pred8)
    pred9=layers.Dense(1,activation='sigmoid',name=y_cols[8])(pred9)
    pred10=layers.Dense(1,activation='sigmoid',name=y_cols[9])(pred10)
    pred11=layers.Dense(1,activation='sigmoid',name=y_cols[10])(pred11)
    pred12=layers.Dense(1,activation='sigmoid',name=y_cols[11])(pred12)
    pred13=layers.Dense(1,activation='sigmoid',name=y_cols[12])(pred13)
    pred14=layers.Dense(1,activation='sigmoid',name=y_cols[13])(pred14)
    pred15=layers.Dense(1,activation='sigmoid',name=y_cols[14])(pred15)
    pred16=layers.Dense(1,activation='sigmoid',name=y_cols[15])(pred16)
    #-----------------------------------------------------------#
    model = Model(posts_input,
                  [pred1,pred2,pred3,pred4,pred5,pred6,
                   pred7,pred8,pred9,pred10,pred11,pred12,
                   pred13,pred14,pred15,pred16])
    #-----------------------------------------------------------#
    model.compile(optimizer='adam',loss='binary_crossentropy')
    model.fit(X,[y[:,i] for i in range(y.shape[1])],epochs=epochs, verbose=False)
    #-----------------------------------------------------------#
    y_hats=model.predict(X)
    for i in range(len(y_hats)):
        predicted_df.iloc[:,i]=y_hats[i]
        
    for thres in thres_list:
        for idx in range(len(y_cols)):
            y_pred_class=(y_hats[idx].flatten()>thres).astype('int')
            score_f1 = f1_score(y[:,idx], y_pred_class)
            score_acc = np.mean((y[:,idx]==y_pred_class).astype('int'))
            df_col='thres'+str(thres)+'_'+str(y_cols[idx])
            performance_df[df_col]=[score_f1,score_acc,0]

    for idx in range(len(y_cols)):
        performance_df.loc['auroc'][performance_df.loc['auroc'].index.str.endswith(y_cols[idx])]=roc_auc_score(y[:,idx].flatten(), y_hats[idx].flatten())
        
    best_cols=[]
    for y_col in y_cols:
        tmp = performance_df[performance_df.columns[performance_df.columns.str.endswith(y_col)]].idxmax(axis=1)['f1']
        best_cols.append(tmp)

    print(performance_df[best_cols].mean(axis=1))