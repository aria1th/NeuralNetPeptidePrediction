###### transformed ######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import swifter
import os
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from copy import deepcopy
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
usrname = os.getlogin()
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pubchempy as pcp
import time

os.getcwd()
# 저랑 같은지 확인해주세요 C:\\Users\\USER\\Desktop\\soup

AbsClock = time.time()

filename = 'C:/Users/%s/Desktop/soup/1.csv'%usrname
import pandas as pd
data_1 = pd.read_csv(filename)
df_1 = pd.DataFrame(data_1)
pd.set_option('display.max_rows', 8000)

class Wrapper(object):
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module_name = module_name

    def __call__(self, *args, **kwargs):
        method = __import__(self.module_name, globals(), locals(), [self.method_name,])
        return method(*args, **kwargs)
    
for i in range(8000):
    new = df_1['Unnamed: 0'][i]
    df_1= df_1.rename(index = {i: new})

df_1 = df_1.rename(index = {0: 'AAA'})
df_1 = df_1.drop('Unnamed: 0', axis = 1)

filename = 'C:/Users/%s/Desktop/soup/2.csv'%usrname
data_2 = pd.read_csv(filename)
df_2 = pd.DataFrame(data_2)
pd.set_option('display.max_rows', 8000)

df_2 = df_2.rename(index = {0: 'ZnO surface'})
df_2 = df_2.drop('Unnamed: 0', axis = 1)
df_2 = df_2.T

filename = 'C:/Users/%s/Desktop/soup/3.csv'%usrname
data_3 = pd.read_csv(filename)
df_3 = pd.DataFrame(data_3)
pd.set_option('display.max_rows', 8000)

df_3 = df_3.rename(index = {0: 'ZnO surface'})
df_3 = df_3.drop('Unnamed: 0', axis = 1)


filename = 'C:/Users/%s/Desktop/soup/4.csv'%usrname
data_4 = pd.read_csv(filename)
df_4 = pd.DataFrame(data_4)
pd.set_option('display.max_rows', 8000)
for i in range(8000):
    new = df_4['Unnamed: 0'][i]
    df_4= df_4.rename(index = {i: new})

df_4 = df_4.rename(index = {0: 'AAA'})
df_4 = df_4.drop('Unnamed: 0', axis = 1)
df_4

## Input of molecular structure

mol_lst = []
for i in range(1,8001):
    mol_name = 'C:/Users/%s/Desktop/soup/protein_structure/ligand_'%usrname+str(i)+'.mol'
    mol = Chem.MolFromMolFile(mol_name)
    mol_lst.append(mol)

obj = {}
df3_T = df_3.T
df3_list = df3_T['ZnO surface'].tolist()


for i in range(37):
    obj['df1_'+str(i)] = df_1[str(i)].tolist()
    obj['df2_'+str(i)] = df_2['ZnO surface'].tolist()
    obj['df3_'+str(i)] = df3_list[i]*8000
    obj['df4_'+str(i)] = df_4[str(i)].tolist()
    
full_df = pd.DataFrame(obj)
full_df

VOC_lst = []
VOC_namelst = []
for i in range(1,38):
    VOC_name = 'C:/Users/%s/Desktop/soup/VOC_structure/'%usrname+str(i)+'.mol'
    VOC = Chem.MolFromMolFile(VOC_name)
    VOC_lst.append(VOC)
    try:
        VOC_namelst.append(pcp.get_compounds(Chem.rdmolfiles.MolToSmiles(VOC), 'smiles')[0].iupac_name)
    except:
        VOC_namelst.append(str(i))
    print('registered VOC : {0}'.format(VOC_namelst[-1]))

len(VOC_lst)
obj = {}
df3_T = df_3.T
df3_list = df3_T['ZnO surface'].tolist()

data_1 = []
data_2 = []
data_3 = []
data_4 = []
mol = []
VOC = []
for i in range(37):
    data_1 += df_1[str(i)].tolist()
    data_2 += df_2['ZnO surface'].tolist()
    data_3 += [df3_list[i]]*8000
    data_4 += df_4[str(i)].tolist()
    mol += mol_lst
    VOC += [VOC_lst[i]]*8000

    
dic = {'mol':mol, 'VOC':VOC, 'pep+VOC':data_1, 'ZnO+pep':data_2, 'ZnO+VOC':data_3, 'ZnO+pep+VOC':data_4}
large_df = pd.DataFrame(dic)
large_df

## Structural Descriptor

m = large_df['mol'][0]
aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
def AromaticAtoms(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i==True:
            aa_count.append(1)
        sum_aa_count = sum(aa_count)
    return sum_aa_count
desc_AromaticAtoms = [AromaticAtoms(element) for element in large_df['mol']]
print('CountAromaticAtoms Done')
#Peptide structural descriptor
import concurrent.futures 
from multiprocessing import cpu_count

def parallelized_apply(target, wrapModule, wrapFunc, cores = 6, parallel = False):
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers = cores) as executor:
            ts = list(executor.map(Wrapper(wrapFunc, wrapModule), target, chunksize = cores))
        print('done work, merging')
        return ts
    else:
        return target.apply(lambda x : wrapFunc(x))
    
def evaluation(model, X_test, y_test, show = False):
    prediction = model(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    RMSE = mse**0.5
    #x_range = range(math.floor(min(len(X_test)), math.ceil(max(len(X_test))+1))
    
    #sorted_prediction = prediction.sort()
    #sorted_test = y_test.sort()
    if show:
        plt.figure(figsize=(15, 10))
        plt.scatter(np.arange(len(prediction)),  (prediction - y_test) / y_test , c='#FA6C00', label="prediction",s=1, alpha = 0.9)
        #plt.scatter(np.arange(len(y_test)), y_test, c='#A1E356', label="actual", s=2, alpha = 0.6)
        #plt.plot(prediction, "o", label="prediction", markerfacecolor = '#54A1BF', markersize = '12', markeredgewidth = '4', markeredgecolor = '#F2CA7E')
        #plt.plot(y_test, 'o', label="actual", color = '#F26F63', markersize = '9')
        plt.legend()
        plt.grid(color = '#BDBDBD', linestyle = '-', linewidth = 0.5)
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.ylabel('Relative Binding Score Difference', fontsize = 20)
        plt.xlabel('X test', fontsize = 20)
        ax = plt.axes()
        ax.invert_yaxis()
        #plt.xticks([0,1,2,3,4])
        #plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
        plt.show()
    
    print('MAE score:', round(mae, 7))
    print('MSE score:', round(mse,7))
    print('RMSE score:', round(RMSE,7))

    
large_df['mol'] =large_df['mol'].apply(lambda x : Chem.AddHs(x))
large_df['VOC'] =large_df['VOC'].apply(lambda x : Chem.AddHs(x))
large_df['Aromatic'] = desc_AromaticAtoms
print('Applied H to mol/VOCs')
from rdkit.Chem import Descriptors as Descriptors
from rdkit.Chem import rdMolDescriptors 
dic = {'mol':mol, 'VOC':VOC, 'pep+VOC':data_1, 'ZnO+pep':data_2, 'ZnO+VOC':data_3, 'ZnO+pep+VOC':data_4, 'aromatic_atoms': desc_AromaticAtoms}
large_df = pd.DataFrame(dic)
FuncNames = ['ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3',
 'HeavyAtomMolWt','MaxAbsPartialCharge','MaxPartialCharge',
             'MinAbsPartialCharge','MinpartialCharge','MolWt','NumRadicalElectrons',
             'NumValenceElectrons']
FuncTarget = {i : j for (i,j) in zip(FuncNames,[Descriptors.ExactMolWt, Descriptors.FpDensityMorgan1, Descriptors.FpDensityMorgan2, Descriptors.FpDensityMorgan3,
              Descriptors.HeavyAtomMolWt, Descriptors.MaxAbsPartialCharge, Descriptors.MaxPartialCharge, Descriptors.MinAbsPartialCharge,
              Descriptors.MinPartialCharge, Descriptors.MolWt, Descriptors.NumRadicalElectrons, Descriptors.NumValenceElectrons])}
FuncTarget['CalcNumAliphaticCarbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles
FuncTarget['CalcNumAromaticCarbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles
FuncTarget['CalcNumAromaticRings'] = rdMolDescriptors.CalcNumAromaticRings

ft = FuncTarget
ft['CalcNumHBA'] = rdMolDescriptors.CalcNumHBA
ft['CalcNumHBD'] = rdMolDescriptors.CalcNumHBD
ft['CalcLabuteASA'] = rdMolDescriptors.CalcLabuteASA
ft['CalcNumRings'] = rdMolDescriptors.CalcNumRings


for funcNames in FuncTarget:
    if funcNames in ['ExactMolWt', 'MaxAbsPartialCharge','MinAbsPartialCharge','HeavyAtomMolWt','FpDensityMorgan3','FpDensityMorgan2']:
        continue
    #test = funcNames
    if FuncTarget[funcNames].__name__ != '<lambda>':
        try:
            large_df['mol_'+funcNames] = parallelized_apply(large_df['mol'],Descriptors, FuncTarget[funcNames])
            print(funcNames + 'Done')
        except:
            print(funcNames + ' failed')
    else:
        large_df['mol_'+funcNames] = large_df['mol'].apply(lambda x : FuncTarget[funcNames](x))
        print(funcNames + 'Done')
    if FuncTarget[funcNames].__name__ != '<lambda>':
        try:
            large_df['VOC_'+funcNames] = parallelized_apply(large_df['VOC'],Descriptors, FuncTarget[funcNames])
            print(funcNames + 'Done')
        except:
            print(funcNames + ' failed')
    else:
        large_df['VOC_'+funcNames] = large_df['VOC'].apply(lambda x : FuncTarget[funcNames](x))
        print(funcNames + 'Done')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


df_test = deepcopy(large_df)

def calculateCompoundFromName(name):
    res = pcp.get_compounds(name, 'name')
    mol = Chem.MolFromSmiles(res[0].canonical_smiles)
    desc_AA = AromaticAtoms(mol)
    mol = Chem.AddHs(mol)
    d = pd.DataFrame()
    for funNames in FuncTarget:
        if funcNames in ['ExactMolWt', 'MaxAbsPartialCharge','MinAbsPartialCharge','HeavyAtomMolWt','FpDensityMorgan3','FpDensityMorgan2']:
            continue
    d['mol'] = mol_lst
    d['VOC'] = [mol] * len(mol_lst)
    #unfinished
     

    
df_test = df_test.drop('mol', axis=1)
df_test = df_test.drop('VOC', axis=1)

minmax_scaler = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler()

y_real = df_test['ZnO+pep+VOC']
df_test = df_test.drop('ZnO+pep+VOC', axis=1) #y값은 미리 빼둔후 제거함

minmax_scaler.fit(df_test)
y_scaler.fit(np.array(y_real).reshape(-1,1))
orig_features = df_test.values
scaled_features = minmax_scaler.transform(df_test)
scaled_result = y_scaler.transform(np.array(y_real).reshape(-1,1))
scaled_result = scaled_result.reshape(scaled_result.shape[0])
maxes = orig_features.max(axis=0)
mins = orig_features.min(axis=0)
scales = maxes - mins #X val scaled


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) 
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
class NeuralNet(nn.Module):
    def __init__(self, n_input_features, hidden_size1, hidden_size2):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(n_input_features, hidden_size1) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out
class NeuralNet3(nn.Module):
    def __init__(self, n_input_features, hidden_size1, hidden_size2, hidden_size3):
        super(NeuralNet3, self).__init__()
        self.linear1 = nn.Linear(n_input_features, hidden_size1) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_size3,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.sigmoid(out)
        return out
    
def gen_dataset(dataset, minmax_scaler = None, y_scaler = None, drop_column = []):
    if minmax_scaler == None:
        minmax_scaler = preprocessing.MinMaxScaler()
        minmax_scaler.fit(dataset)
    if y_scaler == None:
        y_scaler = preprocessing.MinMaxScaler()
        y_scaler.fit(np.array(real_values).reshape(-1,1))
    dataset = deepcopy(dataset)
    for drops in drop_column:
        dataset = dataset.drop(drops, axis = 1)
    dataset = dataset.drop('mol', axis=1).drop('VOC', axis=1)
    real_values = dataset['ZnO+pep+VOC']
    dataset = dataset.drop('ZnO+pep+VOC', axis=1)
    orig_features = dataset.values
    scaled_features = minmax_scaler.transform(dataset)
    scaled_result = y_scaler.transform(np.array(real_values).reshape(-1,1))
    scaled_result = scaled_result.reshape(scaled_result.shape[0])
    return pd.DataFrame(data=scaled_features), scaled_result
def logtest(model, ax, target_data , y_scaler , y_scaled, epoch = 0, loss = 0):
    #train_df, y 주요값과 실제결과값
    with torch.no_grad():
        pre = np.array(model(target_data).cpu()) #predicted data, test = valid dataset
        #예측 y값을 다시 minmaxscaler에서 revert
        predict_values = y_scaler.inverse_transform(pre)
        y_real = y_scaler.inverse_transform(y_scaled.reshape(1,-1).cpu())
        ax.scatter(y_real, predict_values, s=1, alpha=0.6, zorder=1, label = 'epoch : {0} loss : {1}'.format(epoch, loss))
        
def train_dataset(datas, drop = [36],
                    model = NeuralNet, nnl1 = 15, nnl2 = 15, max_iter = 1e4, max_time = 100, criterion = nn.MSELoss, scaler = None, until = 1e-2,
                    manual_epoch_list = [200,400,1000, 40000, 1e5], test_size = 0.3, opt = torch.optim.Adam, VOCnum = 37, PEPnum = 8000, random_seed = 1,
                    drop_column = [], axisMin = -3.75, axisMax = -1.5, external_test = False, nnl3 = False):
    dataset = deepcopy(datas)
    for drops in drop_column:
        dataset = dataset.drop(drops, axis = 1)
    dataset['V'] = sorted(list(range(VOCnum))*PEPnum)
    for drops in drop:
        dataset = dataset[dataset['V'] != drops]
    dataset = dataset.drop('V',axis=1) #process drops
    dataset = dataset.drop('mol', axis=1)
    dataset = dataset.drop('VOC', axis=1)
    minmax_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    real_values = y_real = dataset['ZnO+pep+VOC']
    dataset = dataset.drop('ZnO+pep+VOC', axis=1)
    minmax_scaler.fit(dataset)
    y_scaler.fit(np.array(real_values).reshape(-1,1))
    orig_features = dataset.values
    scaled_features = minmax_scaler.transform(dataset)
    scaled_result = y_scaler.transform(np.array(real_values).reshape(-1,1))
    scaled_result = scaled_result.reshape(scaled_result.shape[0])
    dataset_scaled_VOC = pd.DataFrame(data=scaled_features)
    if nnl3:
        model = NeuralNet3(len(dataset_scaled_VOC.columns), nnl1, nnl2, nnl3)
    else:
        model = model(len(dataset_scaled_VOC.columns), nnl1, nnl2)
    model.cuda()
    X_train, X_test, y_train, y_test = train_test_split(dataset_scaled_VOC, scaled_result, test_size=test_size, random_state=random_seed)
    if scaler != None:
        sc = scaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    if external_test:
        smallfeature, smallresult = gen_dataset(datas, minmax_scaler, y_scaler, drop_column)
        smallfeature, smallresult = torch.from_numpy(np.array(smallfeature,dtype=np.float32)).cuda(), torch.from_numpy(np.array(smallresult,dtype=np.float32)).cuda()
    X_train = torch.from_numpy(X_train.astype(np.float32)).cuda()
    X_test = torch.from_numpy(X_test.astype(np.float32)).cuda()
    y_train = torch.from_numpy(y_train.astype(np.float32)).cuda()
    y_test = torch.from_numpy(y_test.astype(np.float32)).cuda()
    y_train = y_train.view(y_train.shape[0],1)
    y_test = y_test.view(y_test.shape[0],1)
    fig = plt.figure(figsize=(8,8), dpi=80)
    ax = fig.add_subplot(111)
    criterion = criterion()
    optimizer = opt(model.parameters(), lr = 0.01)
    num_epochs = max_iter
    abstime = time.time()
    sttime = time.time()
    epoch = 0
    preloss = 1
    lossdiff = 1
    targetbest = None
    targetloss = 1e6
    minloss = 1
    best = None
    losslist = []
    updatecount = 0
    while lossdiff > until or time.time()-abstime < max_time and epoch < max_iter: #loss change가 일정 수준 이하일 때까지는 시행, 시간 또는 시행 조건 달성시 break
        epoch += 1 
        y_predicted = model(X_train)
        loss = torch.sqrt(criterion(y_predicted, y_train))  #RMSE
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if external_test:
            with torch.no_grad():
                smallpredict = model(smallfeature)
                temploss = torch.sqrt(torch.sum(torch.square(smallpredict.flatten()-smallresult.flatten())))
                if temploss < targetloss:
                    targetbest = deepcopy(model)
                    targetloss = temploss
        losslist.append(math.log10(loss.item()))
        if losslist[-1] < minloss:
            best = deepcopy(model)
            minloss = losslist[-1]
            updatecount += 1
        if epoch in manual_epoch_list:
            logtest(model, ax,X_test, y_scaler, y_test, epoch, loss.item())
        if time.time() - sttime > 10:
            sttime = time.time()
            print(f'epoch : {epoch+1}, loss = {loss.item():.4f}')
            elapsed = sttime - abstime
            lossdiff = (preloss - loss.item()) / preloss
            print('loss difference : {0}'.format(lossdiff))
            preloss = loss.item()
            speed = epoch / elapsed
            eta = min((num_epochs - epoch + 1) / speed, abstime+ max_time - time.time())
            print('test size : {1} ETA: {0}'.format(eta, test_size))
            print('train speed : {0:1f} epoch/s best loss : {1}'.format(speed, 10**minloss))
    print('Updated model to best : {0} times'.format(updatecount))
    with torch.no_grad():
        evaluation(model, X_test, y_test, show=False)
        pre = np.array(model(torch.from_numpy(np.array(dataset_scaled_VOC).astype(np.float32)).cuda()).cpu()) #predicted data, test = valid dataset
        #예측 y값을 다시 minmaxscaler에서 revert
        predicted_values = y_scaler.inverse_transform(pre)
        ax.plot([axisMax, axisMin], [axisMax, axisMin], linestyle=":", c='grey', zorder=2)
        ax.set_ylim(axisMin, axisMax)
        ax.set_xlim(axisMin, axisMax)
        ax.legend(loc='best')
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.ylabel('Predicted Binding Score(kcal/mol)')
        plt.xlabel('Calculated Binding Score(kcal/mol)')
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
    y_reals, predicted_values = np.array(y_real), predicted_values.flatten()
    diff = predicted_values - y_reals
    rel_diff = diff / y_reals
    #plt.violinplot(diff / y_reals)
    #plt.show()
    return best ,losslist, minmax_scaler, y_scaler, ax, rel_diff, targetbest
def testModelBySubset(model, dataset, xScaler, yScaler, subset = [1,2,3], l = [], VOCnum = 37, PEPnum = 8000, label = '', drop_column = [], axisMin = -3.75, axisMax = -1.5):
    if not l:
        epoch = 0
        loss = 0
    else:
        epoch = l.index(min(l))
        loss = min(l)
    dataset = deepcopy(dataset)
    dataset['V'] = sorted(list(range(VOCnum))*PEPnum)
    dataset = dataset[dataset['V'].isin(subset)]
    dataset = dataset.drop('V',axis=1) #Cut out subsets
    td, yscaled = gen_dataset(dataset, xScaler, yScaler, drop_column = drop_column)
    yscaled = torch.from_numpy(yscaled.astype(np.float32)).cuda()
    fig = plt.figure(figsize=(8,8), dpi=80)
    ax = fig.add_subplot(111)
    logtest(model, ax, epoch = epoch, loss = 10**loss, target_data = torch.from_numpy(np.array(td,dtype=np.float32)).cuda(), y_scaler =yScaler, y_scaled = yscaled)
    ax.plot([axisMax, axisMin], [axisMax, axisMin], linestyle=":", c='grey', zorder=2)
    ax.set_ylim(axisMin, axisMax)
    ax.set_xlim(axisMin, axisMax)
    ax.legend(loc='best')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Predicted Binding Score(kcal/mol)')
    plt.xlabel('Calculated Binding Score(kcal/mol)')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    ax.set_title(label)
    plt.show()
    


    

#Perform a train-test split. 비율은 제가 임의로 20%로 정했습니다. 데이터가 충분히 크기 때문입니다.
def test(test_size=0.3, opt = torch.optim.Adam, model = None, max_iter = 1e4, max_time = 100, 
         criterion = nn.MSELoss, scaler = None, until = 1e-4, manual_epoch_list = [200,400,1000, 40000, 1e5], target_data = None, scaled_data = None):
    X_train, X_test, y_train, y_test = train_test_split(target_data, scaled_data, test_size=test_size, random_state=1) #model = NeuralNet(len(train_df.columns), 15, 15)
    if scaler != None:
        sc = scaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    losslist = list()
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    #bceLoss = nn.BCELoss()

    y_train = y_train.view(y_train.shape[0],1)
    y_test = y_test.view(y_test.shape[0],1)
    fig = plt.figure(figsize=(8,8), dpi=80)
    ax = fig.add_subplot(111)
    criterion = criterion()
    optimizer = opt(model.parameters(), lr = 0.01)
    num_epochs = max_iter
    abstime = time.time()
    sttime = time.time()
    epoch = 0
    preloss = 1
    lossdiff = 1
    minloss = 1
    best = None
    updatecount = 0
    while lossdiff > until or time.time()-abstime < max_time and epoch < max_iter: #loss change가 일정 수준 이하일 때까지는 시행, 시간 또는 시행 조건 달성시 break
        epoch += 1 
        y_predicted = model(X_train)
        loss = torch.sqrt(criterion(y_predicted, y_train))  #RMSE
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losslist.append(math.log10(loss.item()))
        if losslist[-1] < minloss:
            best = deepcopy(model)
            minloss = losslist[-1]
            updatecount += 1
        if epoch in manual_epoch_list:
            logtest(model, ax,train_df, y_scaler, y_real, epoch = epoch, loss = loss.item())
        if time.time() - sttime > 10:
            sttime = time.time()
            print(f'epoch : {epoch+1}, loss = {loss.item():.4f}')
            elapsed = sttime - abstime
            lossdiff = (preloss - loss.item()) / preloss
            print('loss difference : {0}'.format(lossdiff))
            preloss = loss.item()
            speed = epoch / elapsed
            eta = (num_epochs - epoch + 1) / speed
            print('test size : {1} ETA: {0}'.format(eta, test_size))
    print('Updated model to best : {0} times'.format(updatecount))
    with torch.no_grad():
        evaluation(model, X_test, y_test, show=False)
    return model, ax, losslist, best
    
def evaluation(model, X_test, y_test, show = False):
    prediction = model(X_test)
    mae = mean_absolute_error(y_test.cpu(), prediction.cpu())
    mse = mean_squared_error(y_test.cpu(), prediction.cpu())
    RMSE = mse**0.5
    #x_range = range(math.floor(min(len(X_test)), math.ceil(max(len(X_test))+1))
    
    #sorted_prediction = prediction.sort()
    #sorted_test = y_test.sort()
    if show:
        plt.figure(figsize=(15, 10))
        plt.scatter(np.arange(len(prediction)),  ((prediction - y_test) / y_test).cpu() , c='#FA6C00', label="prediction",s=1, alpha = 0.9)
        #plt.scatter(np.arange(len(y_test)), y_test, c='#A1E356', label="actual", s=2, alpha = 0.6)
        #plt.plot(prediction, "o", label="prediction", markerfacecolor = '#54A1BF', markersize = '12', markeredgewidth = '4', markeredgecolor = '#F2CA7E')
        #plt.plot(y_test, 'o', label="actual", color = '#F26F63', markersize = '9')
        plt.legend()
        plt.grid(color = '#BDBDBD', linestyle = '-', linewidth = 0.5)
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.ylabel('Relative Binding Score Difference', fontsize = 20)
        plt.xlabel('X test', fontsize = 20)
        ax = plt.axes()
        ax.invert_yaxis()
        #plt.xticks([0,1,2,3,4])
        #plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
        plt.show()
    
    print('MAE score:', round(mae, 7))
    print('MSE score:', round(mse,7))
    print('RMSE score:', round(RMSE,7))
    
#여기서 샵 표시된 코드를 activate하면 다른 그래프가 나와용! 원하시는 걸 사용하세요.

def aftertest(model, ax, do_again = False, optional = '<>', target_data = None, y_real = None, y_scaler = None):
    #train_df, y 주요값과 실제결과값
    with torch.no_grad():
        pre = np.array(model(torch.from_numpy(np.array(target_data).astype(np.float32))).cpu()) #predicted data, test = valid dataset
        #예측 y값을 다시 minmaxscaler에서 revert
        predict_values = y_scaler.inverse_transform(pre)
        if do_again:
            fig = plt.figure(figsize=(8,8), dpi=80)
            ax = fig.add_subplot(111)
            ax.scatter(y_real, predict_values, s=1, c="orange", label = 'best at {0}'.format(optional), alpha=0.6, zorder=1)
        ax.plot([-1.5, -3.75], [-1.5, -3.75], linestyle=":", c='grey', zorder=2)
        ax.set_ylim(-3.75, -1.5)
        ax.set_xlim(-3.75, -1.5)
        ax.legend(loc='best')
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.ylabel('Predicted Binding Score(kcal/mol)')
        plt.xlabel('Calculated Binding Score(kcal/mol)')
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        plt.show()
    return y_real, predict_values


'''
df_scaled_VOC = pd.DataFrame(data=scaled_features)
y = scaled_result
train_df = df_scaled_VOC
print('Start training')
mod, ax, losslist, best = test(until = 1e-2, max_iter = 1e4, max_time = 30000)
print('Finished training based on loss change rate')
print('Elapsed time : {0}'.format(time.time() - AbsClock))
y_reals, predicted_values = aftertest(mod, ax)
y_reals, predicted_values = np.array(y_reals), predicted_values.flatten()
diff = predicted_values - y_reals
plt.violinplot(diff / y_reals)
plt.show() #show probablity regard to value relative differences (how bigger than real values?)'''


        
def getPredict(dataset, model, x_scaler, y_scaler, drop_column = [], counts = 5, printout = False, returnasValue = False):
    readable = deepcopy(dataset)
    selectedSymbol = 'LITSFANMPGKQYVHWDERC'
    genSelectedTripeptide = sorted(''.join([i,j,k]) for i in selectedSymbol for j in selectedSymbol
                                     for k in selectedSymbol)
    readable['pepName'] = genSelectedTripeptide*37 #Naming 과정이며 Name 자체는 중요하지 않을 수 있으므로 대체 가능
    readable['VOCName'] = sorted(list(range(37))*8000) #마찬가지로 VOC naming 과정
    target_data, scaled_result = gen_dataset(dataset, x_scaler, y_scaler, drop_column = drop_column)
    with torch.no_grad():
        pre = model(torch.from_numpy(np.array(target_data,dtype=np.float32)).cuda()) #predicted data, test = valid dataset
    predicted_values = y_scaler.inverse_transform(pre.cpu()).flatten()
    readable['predicted'] = predicted_values #모델에 의한 예측값
    if returnasValue : return predicted_values
    result_view = readable[['VOCName','pepName','ZnO+pep+VOC', 'predicted']]
    predictResult = {}
    realResult = {}
    for vocCode, realName in enumerate(VOC_namelst):
        predictResult[realName] = result_view['pepName'][np.where(result_view[result_view['VOCName'] == vocCode]['predicted'].isin ((result_view[result_view['VOCName'] == vocCode]['predicted']).nsmallest(counts+1).values))[0]].values.tolist()
        realResult[realName] = result_view['pepName'][np.where(result_view[result_view['VOCName'] == vocCode]['ZnO+pep+VOC'].isin ((result_view[result_view['VOCName'] == vocCode]['ZnO+pep+VOC']).nsmallest(counts+1).values))[0]].values.tolist()

    for i in predictResult:
        if printout:
            print('{0}'.format(i).ljust(40) +  '{0}'.format(predictResult[i]))#pretty format
    return predictResult, realResult
'''
y_reals, y_best = aftertest(best, ax, True, 'epoch : {0}, loss : {1}'.format(losslist.index(min(losslist)), 10**min(losslist)))
y_reals, y_best = np.array(y_reals), y_best.flatten()
diff = y_best - y_reals
plt.violinplot(diff / y_reals)
plt.show() #show violin plot of best result'''

print('Finished preparing data')
test_all = True
drop = [36]
max_time = 3600
max_iter = 1e5

if test_all:
    b,l, xscale, yscale , ax, rel_diff, tb = train_dataset(large_df, drop = drop, max_iter = 1e5, max_time = 150, manual_epoch_list = [1000, 20000, 200000, 400000], external_test = True, nnl1 = 30)
    ax.set_title('Trained without VOC : {0}'.format(str(drop)))
    plt.show() #perform training by flawed dataset(missing voc)
    plt.violinplot(rel_diff)
    plt.show()
    testModelBySubset(b,large_df,xscale,yscale,drop,l, label = 'prediction of VOC :{0}'.format(str(drop))) #VOC Code based

drop_column = ['pep+VOC'] #WHY
b2,l2, xscale2, yscale2 , ax2, rel_diff2, tb2 = train_dataset(large_df, drop = drop, max_iter = max_iter, max_time = max_time, drop_column = drop_column, manual_epoch_list = [1000, 20000, 200000, 400000], external_test = True, nnl1 = 30)
ax2.set_title('Trained without VOC : {0} without potential'.format(str(drop)))
plt.show()#perform training without potential data
plt.violinplot(rel_diff2)
plt.show()
testModelBySubset(b2,large_df,xscale2,yscale2,list(range(37)),l2, label = 'prediction without potential', drop_column = drop_column ) #best trained
testModelBySubset(b2,large_df,xscale2,yscale2,drop,l2, label = 'prediction of VOC :{0} without potential{1}'.format(str(drop), str(drop_column)), drop_column = drop_column )#best split toward external data
predicted, reals = getPredict(large_df, b2, xscale2, yscale2, drop_column = drop_column)

def csvoutput(dictionary):
    for keys in dictionary:
        print(keys+'`'+ '`'.join(dictionary[keys]))

def histogramInPlace(hist, pepstr):
    for i in pepstr:
        if i in hist:
            hist[i] += 1
        else:
            hist[i] = 1
    return hist
def histogramMerge(listed):
    hist = dict()
    for i in listed:
        hist = histogramInPlace(hist, i)
    return hist

def processPrediction(dictionary):
    newdict = dict()
    for keys in dictionary:
        newdict[keys] = histogramMerge(dictionary[keys])
    return newdict

def statisticPrediction(dictionary, setKeys = ['W']):
    #'W' percentage
    newdict = dict()
    for keys in dictionary:
        newdict[keys] = sum(dictionary[keys][i] / sum(dictionary[keys].values()) for i in setKeys)
    return newdict
        
statistic = statisticPrediction(processPrediction(predicted),setKeys = 'WYP')
