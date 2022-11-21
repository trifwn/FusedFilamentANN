import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

def preprocess(df):
    # Change Strings To nums
    df['grid'] = df['infill_pattern'].replace(['grid','honeycomb'], ['1','0']).astype(int)
    df['honeycomb'] = df['infill_pattern'].replace(['grid','honeycomb'], ['0','1']).astype(int)
    junk = df.pop('infill_pattern')

    df['pla'] = df['material'].replace(['pla','abs'], ['0','1']).astype(int)
    df['abs'] = df['material'].replace(['pla','abs'], ['1','0']).astype(int)
    junk = df.pop('material')

    # Normalize data
    cols_to_norm = ['layer_height (mm)', 'wall_thickness (mm)', 'infill_density (%)',
            'nozzle_temperature (0C)', 'bed_temperature (0C)',
            'print_speed (mm/s)', 'fan_speed (%)']

    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: -1 + 2*(x - x.min()) / (x.max() - x.min())).astype(np.float32)
    return df 

def getKmeansPoints(data,idx,n):
    kmeans = KMeans(n_clusters=n,n_init=10,random_state=10)
    data['label'] = kmeans.fit_predict(data[[idx]])

    ord_idx=np.argsort(kmeans.cluster_centers_.flatten())

    cntrs = np.zeros_like(data['label'])-1
    for i in np.arange(n):
        cntrs[data['label']==ord_idx[i]]=i
    data['label'] = cntrs
    idx =[]
    for i in np.arange(n):
        idx.append(*data[data['label']==i].sample(1).index.values)
    return idx

def kMeansSplit(df,index):
    folders = ['Rough','UTS','Elon']
    target = ['roughness (μm)','tension_strength (MPa)','elongation (%)']

    CASE = folders[index]
    testidc = getKmeansPoints(df,target[index],7)
    test = df.loc[testidc]
    df = df.drop(axis=1,index=testidc)

    validc =  getKmeansPoints(df,target[index],7)
    validate = df.loc[validc]
    train = df.drop(axis=1,index=validc)

    train_indc = train.index.sort_values()
    val_indc = validate.index.sort_values()
    test_indc = test.index.sort_values()

    # Save indexes
    np.savetxt(os.path.join(CASE,'train.dat'),train_indc, fmt='%i')
    np.savetxt(os.path.join(CASE,'test.dat'),test_indc, fmt='%i')
    np.savetxt(os.path.join(CASE,'val.dat'),val_indc, fmt='%i')


    return train,validate,test

def getParameters(index):
    # Choose Data to Train
    cols2go = []
    cols2go.append(['print_speed (mm/s)','fan_speed (%)','wall_thickness (mm)',
        'pla','abs','nozzle_temperature (0C)','layer_height (mm)'])
    # ROUGHNESS = Print Speed Boroume kai oxi

    cols2go.append(['fan_speed (%)','print_speed (mm/s)','pla','abs','layer_height (mm)',
                'infill_density (%)','wall_thickness (mm)','nozzle_temperature (0C)'])
    # UTS = AFAIRESI PRINT SPEED

    cols2go.append(['fan_speed (%)','pla','abs','layer_height (mm)','nozzle_temperature (0C)',
                    'wall_thickness (mm)','infill_density (%)'])
    # ELON = layerHeight NozzleTemp   
    return cols2go[index]

def importAndProcess(index):
    target = ['roughness (μm)','tension_strength (MPa)','elongation (%)']
    # Input Data
    df = pd.read_csv('data.csv')

    # Serialize
    df =df.sort_values(target[index]).reset_index()
    df.pop('index')

    # Normalize 
    df = preprocess(df)

    # Split Data
    train , validate, test = kMeansSplit(df,index)

    # Categorize x and y
    ytrain = train.pop(target[index])
    yval =validate.pop(target[index])
    ytest = test.pop(target[index])

    cols = getParameters(index)
    xtrain = train[(i for i in cols)]
    xval = validate[(i for i in cols)]
    xtest = test[(i for i in cols)]
    return xtrain,xval,xtest,ytrain,yval,ytest

def importForK(index):
    target = ['roughness (μm)','tension_strength (MPa)','elongation (%)']

    # Input Data
    df = pd.read_csv('data.csv')

    # Serialize
    df =df.sort_values(target[index]).reset_index()
    df.pop('index')
    
    # Normalize
    df = preprocess(df)
    x = df.sample(frac=1,random_state=42)
    y=  df[target[index]]
    
    # Choose Data to Train
    cols = getParameters(index)
    x = x[(i for i in cols)]

    return x,y

def LoadAndProcess(index):
    folders = ['Rough','UTS','Elon']
    target = ['roughness (μm)','tension_strength (MPa)','elongation (%)']
    CASE = folders[index]
    # Input Data
    df = pd.read_csv('data.csv')

    # Serialize
    df =df.sort_values(target[index]).reset_index()
    df.pop('index')

    # Normalize
    df = preprocess(df)

    # Split Data
    train_indc = np.loadtxt(os.path.join(CASE,'train.dat'),dtype=int)
    test_indc = np.loadtxt(os.path.join(CASE,'test.dat'),dtype=int)
    val_indc = np.loadtxt(os.path.join(CASE,'val.dat'),dtype=int)
    
    train = df.loc[train_indc].sort_values(target[index])
    validate = df.loc[val_indc].sort_values(target[index])
    test = df.loc[test_indc].sort_values(target[index])

    ytrain = train.pop(target[index])
    yval =validate.pop(target[index])
    ytest = test.pop(target[index])

    # Choose Data to Train
    cols = getParameters(index)
    xtrain = train[(i for i in cols)]
    xval = validate[(i for i in cols)]
    xtest = test[(i for i in cols)]

    return xtrain,xval,xtest,ytrain,yval,ytest

# class WeightCapture(tf.keras.callbacks.Callback):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.weights = []
#         self.epochs = []
        
#     def on_epoch_end(self, epoch, logs=None):
#         self.epochs.append(epoch) # remember the epoch axis
#         weight = {}
#         for layer in self.model.layers:
#             if not layer.weights:
#                 continue
#             name = layer.weights[0].name.split("/")[0]
#             weight[name] = layer.weights[0].numpy()
#         self.weights.append(weight)