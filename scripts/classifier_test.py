#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:17:23 2019

@author: jsearcy

A small classifier example


"""
from vydata import VyData
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pdb
from sklearn.metrics import confusion_matrix
import argparse
import xgboost as xgb
import shap
os.environ['CUDA_VISIBLE_DEVICES']=''


def LassoClassifier(input_size,output_size):
    input_v=tf.keras.Input( (input_size,) )
    output=tf.keras.layers.Dense(output_size,kernel_constraint=tf.keras.constraints.UnitNorm(),use_bias=False, activation='softmax')(input_v) 
    model=tf.keras.models.Model(input_v,output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    grads=[tf.gradients(model.output[:,i],model.inputs)[0] for i in range(output_size)]

    return model,grads




def Classifier(input_size,output_size):
    input_v=tf.keras.Input( (input_size,) )
    nn=tf.keras.layers.Dense(65)(input_v) #Massive data reduction here !
    nn=tf.keras.layers.LeakyReLU()(nn)
    nn=tf.keras.layers.Dropout(0.8)(nn)

    nn=tf.keras.layers.Dense(65)(nn) #Massive data reduction here !

    nn=tf.keras.layers.LeakyReLU()(nn)
    nn=tf.keras.layers.Dropout(0.8)(nn)
    nn_output=tf.keras.layers.Dense(output_size)(nn)     

    output=tf.keras.layers.Activation(activation='softmax')(nn_output)     

    model=tf.keras.models.Model(input_v,output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    grads=[tf.gradients(nn_output[:,i],model.inputs)[0] for i in range(output_size)]
    return model,grads

def xgboost(input_size,output_size):
    model = xgb.XGBClassifier(n_estimators=20,max_depth=5)
    return model

def historyplot(history,name):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    
    f=plt.figure()
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    f.savefig(name)
    plt.close(f)

def plot_confusion_matrix(true,pred,weights,var_list,name):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    
#    pred=np.argmax(ypred,axis=1)
#    true=np.argmax(ytrue,axis=1)
    cm=np.zeros((len(var_list),len(var_list)))
    for t,p,w in zip(true,pred,weights):
        cm[t,p]+=1
#    pdb.set_trace()
    print(cm)
        
    total=np.sum(cm,axis=1)
    for i,v in enumerate(total):
        cm[i,:]=cm[i,:]/v


  #  cm=confusion_matrix(true,pred,sample_weight=weights)/total

    fig, ax = plt.subplots(figsize=(5,5))
   
    ax.set_xticks(np.arange(len(cm)))
    ax.set_yticks(np.arange(len(cm)))

    ax.set_xticklabels(var_list,rotation=90)
    ax.set_yticklabels(var_list)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    im =  ax.imshow(cm,vmin=0.0,vmax=1.0)
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, str(round(cm[i, j]*100))+'%',
                       ha="center", va="center", color="w")
    
    fig.savefig(name,bbox_inches="tight")
    plt.show()    
    plt.close(fig)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
   
    parser.add_argument('--threshold', dest='threshold', action='store',
                        default=50,type=int)
    parser.add_argument('--target', dest='target', action='store',
                         default='Management')
    parser.add_argument('--type', dest='type',action='store',default='xgb')
    parser.add_argument('--nobalance', dest='nobalance',action='store_true',default=False)
    parser.add_argument('--input_data', dest='input_data',action='store',default='')
    parser.add_argument('--input_metadata', dest='input_md',action='store',default='')
    parser.add_argument('--output', dest='output',action='store',default='')

    

    args=parser.parse_args()
    vyd=VyData(args.threshold,data_location=args.input_data,meta_data_location=args.input_md)
    plot_out=os.path.join(args.output,'plots')

    y=vyd.get_one_hot(args.target)

    output_size=y.shape[1]
    class_ids=np.argmax(y,axis=1)
    
    if not args.nobalance:
        class_weights={}
        for i in range(len(vyd.variable_lists[args.target])):
            class_weights[i]=np.sum(y)/np.sum(class_ids==i)
        sample_weights=np.array([class_weights[i] for i in class_ids])
    else:
        sample_weights=np.ones(len(y))


    valid=[]
    train=[]        
      
    for i in range(len(y)):
        if np.random.uniform() > 0.7:valid.append(i)
        else:train.append(i)
            
    data=np.array(vyd.df)
    if args.type=='mlp':model,grads=Classifier(data.shape[1],output_size)        
    if args.type=='linear':model,grads=LassoClassifier(data.shape[1],output_size)
    if args.type=='xgb':model=xgboost(data.shape[1],output_size)

    if args.type in ['linear','mlp']:
        s=tf.keras.backend.get_session()
        es=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')
        history=model.fit(data[train],y[train],
                          validation_data=[data[valid],y[valid],sample_weights[valid]],epochs=300,sample_weight=sample_weights[train],callbacks=[es])
    #       
        g_vals=s.run(grads,feed_dict={model.input:data})
    
        historyplot(history,plot_out+'/'+args.target+"_history.png")
    
        pred=np.argmax(model.predict(data[valid]),axis=1)
        model.save_weights(args.output+'/model.h5')
    
        
        
    
        
        imp_vec=[np.sum(g_vals[i],axis=0) for i in range(output_size)]
        imp=np.argsort(imp_vec[3])
    if args.type=='linear':
         weight_matrix=model.get_weights()[0]
         for i,v in enumerate(vyd.variable_lists[args.target]):
             index=list(np.argsort(weight_matrix[:,i]))
             print('------Most Indicating for not '+v)
             
             for q in index[0:5]:
                 print(vyd.vocab_list[q],q,weight_matrix[q,i])
             index.reverse()
             
             print('------Most Indicating for '+v)
             for q in index[0:5]:
                 print(vyd.vocab_list[q],q,weight_matrix[q,i])


        
        
    elif args.type=='xgb':
        
        #dtrain = xgb.DMatrix(data[train], label=class_ids[train], feature_names=vyd.vocab_list)
        #dtest = xgb.DMatrix(data[valid], label=class_ids[valid], feature_names=vyd.vocab_list)
        
        fit_model=model.fit(data[train],class_ids[train], eval_metric="merror", eval_set=[(data[valid],class_ids[valid])],early_stopping_rounds=10, verbose=True )
        pred=fit_model.predict(data[valid])
        top_n=np.argsort(fit_model.feature_importances_)[-10:]
        fit_model.save_model(plot_out+'/'+args.target+'_xgb.model')
        print('Top 10 Most important Species')
        for i in top_n:
            print(i,vyd.vocab_list[i])
        
        explainer = shap.TreeExplainer(fit_model)
        shap_values = explainer.shap_values(data)

        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        
        for i in vyd.variable_lists[args.target]:
            print(i)
            f=plt.figure()
            index=vyd.variable_lists[args.target][i]            
            shap.summary_plot(shap_values[index],vyd.df)
            f.savefig(plot_out+'/shap_'+i+'.png',bbox_inches='tight',dpi=600)
        
    plot_confusion_matrix(class_ids[valid],pred,sample_weights[valid],vyd.variable_lists[args.target],plot_out+'/'+args.target+"_confusion.png")
    
