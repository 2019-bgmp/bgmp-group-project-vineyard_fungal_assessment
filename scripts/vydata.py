#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:56:05 2019

@author: jsearcy
"""
import numpy as np
import pandas as pd
import pdb
class VyData():
    def __init__(self,threshold=50,data_location='/home/jsearcy/vineyard/VineyardFiles/Summer/VMP-U18_fungiASVs.txt'
                 ,meta_data_location='/home/jsearcy/vineyard/VineyardFiles/Summer/VMP-U18metaFixed.csv'):
        self.md=  pd.read_csv(meta_data_location,sep=',',header=(0),index_col='Sample ID',dtype=str).replace(np.nan, 'Missing', regex=True)
        self.df = pd.read_csv(data_location,sep='\t',header=(0)).replace(np.nan, '?', regex=True)
        vcounts=np.sum(self.df!=0,axis=1)
        ###Number of points a species must exist in to be considered

        self.s_list=['-'.join(i[1].tolist()+[str(index)]) for index,i in enumerate(self.df[['Family','Genus','Species']].iterrows())]
        c_name={}    
        for i,v in enumerate(self.s_list):
            c_name[i]=v
        self.df.rename(index=c_name,inplace=True)

        drop_i=[c_name[i] for i,v in enumerate(vcounts) if v < threshold]

        self.df=self.df.drop(drop_i,axis=0)
        
        self.vocab_list=['-'.join(i[1].tolist()+[str(index)]) for index,i in enumerate(self.df[['Kingdom','Phylum','Class','Order','Family','Genus','Species']].iterrows())]

        self.threshold=threshold

        self.samples=[i.strip('\n') for i in  list(self.md.index) if i.strip('\n') in self.df]
        self.missing_samples=[i for i in  list(self.md.index) if i.strip('\n') not in self.df]
        self.md=self.md.drop(self.missing_samples)
        
        self.variable_lists={}
        for col in self.md.keys():
            self.variable_lists[col]={} #This is a dictionary for quick one-hot encoding
            for i,v in enumerate(self.md[col].unique()):#Should probably save these to keep them consistent
            
                self.variable_lists[col][v]=i
        print(self.df.shape)
        self.df=self.df[self.samples].T  

#        self.data=pd.concat([self.df,self.md],axis='Samples')
    def get_one_hot(self,var):
        data=np.zeros(  (len(self.md),len(self.variable_lists[var])) )
        for i,v in enumerate(self.md[var]):
            data[i, self.variable_lists[var][v]  ]=1
        return data
    
    def get_sequences(self):
        sequences=[]
        for i in range(len(self.df)):
            seq=[ (v,i) for i,v in enumerate(self.df.iloc[i]) if v !=0]
            seq.sort(reverse=True)
            sequences.append([i[1]+1 for i in seq]) #save 0 for end or not in list
        return sequences
    
    
    
    
    def _ut_size(self):
        if self.md.shape[0] != self.df.shape[0]:
            raise(Exception('Unit Test Failed Meta-Data and Data Frames have Different Sizes'))

    def _ut_1hot(self):
        data=self.get_one_hot(self.md.keys()[0])
        
        if self.md.shape[0] != data.shape[0]:
            raise(Exception('Unit Test Failed 1-hot encoding and Data Frames have Different Sizes'))

        if np.sum(data) != data.shape[0]:
            raise(Exception('Unit Test Failed 1-hot encoding has more or less 1s than data points'))



if __name__=='__main__':
    vd=VyData()
    vd._ut_size()
    vd._ut_1hot()
    

        
