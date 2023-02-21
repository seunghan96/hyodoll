import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, roc_auc_score

DATA_DIR = '/Users/LSH/Desktop/hyodoll/data'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 200, help = '# of epochs')

args = parser.parse_args()

epochs = args.epochs

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.shared_layer = nn.Linear(input_size, 32)
        self.month_shared_layer = nn.Linear(32, 64)
        self.time_shared_layer = nn.Linear(32, 64)
        self.time_layer1 = nn.ModuleList([nn.Linear(64, 8) for _ in range(6)])
        self.month_layer1 = nn.ModuleList([nn.Linear(64, 8) for _ in range(12)])
        self.time_layer2 = nn.ModuleList([nn.Linear(8, 1) for _ in range(6)])
        self.month_layer2 = nn.ModuleList([nn.Linear(8, 1) for _ in range(12)])
        
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.shared_layer(x))
        x_month = self.activation(self.month_shared_layer(x))
        x_time = self.activation(self.time_shared_layer(x))
        
        time_out_list = []
        month_out_list = []
        
        for layer in self.time_layer1:
            out = layer(x_time)
            time_out_list.append(self.activation(out))
        
        for layer in self.month_layer1:
            out = layer(x_month)
            month_out_list.append(self.activation(out))
        
        time_final_pred = []
        month_final_pred = []
        for layer, out in zip(self.time_layer2, time_out_list):
            time_final_pred.append(self.sigmoid(layer(out)))
            
        for layer, out in zip(self.month_layer2, month_out_list):
            month_final_pred.append(self.sigmoid(layer(out)))
            
        return month_final_pred + time_final_pred

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(DATA_DIR,'cluster_df_merged.csv'))
    dummy_col = pd.get_dummies(data['cluster'])
    dummy_col.columns = ['cluster_'+str(i) for i in dummy_col.columns]
    data = pd.concat([data,dummy_col],axis=1)

    X_cols = [x for x in data.columns if 'FA_' in x] + [x for x in data.columns if 'cluster_' in x]
    y_cols = [x for x in data.columns if 'emergency_month' in x] + [x for x in data.columns if 'emergency_time' in x]
    X = data[X_cols].values 
    
    thres_list = [0.2, 0.4, 0.6, 0.8]
    predicted_df = pd.DataFrame(index=data.index, columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    performance_index = ['f1', 'acc', 'auroc']
    performance_df = pd.DataFrame(index=performance_index, columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])

    X = torch.tensor(data[X_cols].values).float()#.cuda()
    y = torch.tensor(data[y_cols].values).float()#.cuda()
    y[y > 0] = 1
    
    model = Net(X.shape[1])#.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hats = model(X)
        loss = 0
        for i in range(y.shape[1]):
            loss += criterion(y_hats[i], y[:, i].unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    y_hats = [i.detach().cpu().numpy() for i in y_hats]
    
    for i in range(len(y_hats)):
        predicted_df.iloc[:,i]=y_hats[i]
        
    for thres in thres_list:
        for idx in range(len(y_cols)):
            y_pred_class=(y_hats[idx].flatten()>thres).astype('int')
            score_f1 = f1_score(y.detach().cpu().numpy()[:,idx], y_pred_class)
            score_acc = np.mean((y.detach().cpu().numpy()[:,idx]==y_pred_class).astype('int'))
            df_col='thres'+str(thres)+'_'+str(y_cols[idx])
            performance_df[df_col]=[score_f1,score_acc,0]

    for idx in range(len(y_cols)):
        performance_df.loc['auroc'][performance_df.loc['auroc'].index.str.endswith(y_cols[idx])]=roc_auc_score(y[:,idx].flatten(), y_hats[idx].flatten())
        
    best_cols=[]
    for y_col in y_cols:
        tmp = performance_df[performance_df.columns[performance_df.columns.str.endswith(y_col)]].idxmax(axis=1)['f1']
        best_cols.append(tmp)

    print(performance_df[best_cols].mean(axis=1))    