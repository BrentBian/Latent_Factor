# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import linalg
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import timeit
from collections import defaultdict

class LatentFactor:
    def __init__(self, train, test, d, method):
        
        self.method = method
        self.test = test
        self.train = train
        self.d = d
        
        if self.method == 'pool':
            self.train_user_key = defaultdict(list)
            self.test_user_key = defaultdict(list)
            self.train_joke_key = defaultdict(list)
            self.u = {}
            self.v = {}
            
            print('Initializing, please wait', end='')
            count = 0
            togo = len(train) + len(test)
            interval = togo // 10
            
            for j in train.iterrows():
                i = j[1].tolist()
                self.train_user_key[i[0]].append((i[1], i[2]))
                self.train_joke_key[i[1]].append((i[0], i[2]))
                count += 1
                if count % interval == 0:
                    print('.', end ='')
                
            for j in test.iterrows():
                i = j[1].tolist()
                self.test_user_key[i[0]].append((i[1], i[2]))
                count += 1
                if count % interval == 0:
                    print('.', end ='')
            print()
        
        elif self.method == 'SVD':
            
            self.train_user_key = defaultdict(list)
            self.test_user_key = defaultdict(list)
            self.train_joke_key = defaultdict(list)
            self.u = {}
            self.v = {}
            self.table = np.zeros((24983,100))
            print('Initializing, please wait', end='')
            count = 0
            togo = len(train) + len(test)
            interval = togo // 10
            
            for j in train.iterrows():
                i = j[1].tolist()
                self.train_user_key[i[0]].append((i[1], i[2]))
                self.train_joke_key[i[1]].append((i[0], i[2]))
                self.table[int(i[0])-1,int(i[1])-1] = i[2]
                count += 1
                if count % interval == 0:
                    print('.', end ='')
                
            for j in test.iterrows():
                i = j[1].tolist()
                self.test_user_key[i[0]].append((i[1], i[2]))
                count += 1
                if count % interval == 0:
                    print('.', end ='')
            print()
        
        elif self.method == 'AM':
            
            self.train_user_key = defaultdict(list)
            self.test_user_key = defaultdict(list)
            self.train_joke_key = defaultdict(list)
            
            self.train_user_reviews = defaultdict(list)
            self.train_joke_reviews = defaultdict(list)
            
            
            self.u = {}
            self.v = {}
            self.lamb = 5
            
            print('Initializing, please wait', end='')
            count = 0
            togo = len(train) + len(test)
            interval = togo // 10
            
            for j in train.iterrows():
                i = j[1].tolist()
                self.train_user_key[i[0]].append((i[1], i[2]))
                self.train_joke_key[i[1]].append((i[0], i[2]))
                
                self.train_user_reviews[i[0]].append(i[2])
                self.train_joke_reviews[i[1]].append(i[2])
                
                count += 1
                if count % interval == 0:
                    print('.', end ='')
            
            
            for j in test.iterrows():
                i = j[1].tolist()
                self.test_user_key[i[0]].append((i[1], i[2]))
                count += 1
                if count % interval == 0:
                    print('.', end ='')
            print()
            
    
            
            
            
    def MSE(self, target, new_data = None):
        
        res = 0
        
        if target == 'test':
        
            for j in self.test.iterrows():
                i = j[1].tolist()
                u_id = i[0]
                j_id = i[1]
                r = i[2]
                
                res += (self.u[u_id].T.dot(self.v[j_id]) - r)**2
                
            print('MSE for test set is {:.4f}'.format( res/len(self.test)))
            return res/len(self.test)
        elif target == 'train':
            for j in self.train.iterrows():
                i = j[1].tolist()
                u_id = i[0]
                j_id = i[1]
                r = i[2]
                
                res += (self.u[u_id].T.dot(self.v[j_id]) - r)**2
                
            print('MSE for training set is {:.4f}'.format( res/len(self.train)))
            return res/len(self.train)
        elif target == 'cv':
            for j in new_data.iterrows():
                i = j[1].tolist()
                u_id = i[0]
                j_id = i[1]
                r = i[2]
                
                res += (self.u[u_id].T.dot(self.v[j_id]) - r)**2
            
            print('MSE for validation set is {:.4f}'.format( res/len(new_data)))
            return res/len(new_data)
                
    
    def MAE(self, target):
        
        res = 0
        
        if target == 'test':
            for i in self.test_user_key.keys():
                tmp = 0
                reviews = self.test_user_key[i]
                for j in reviews:
                    u_id = i
                    j_id = j[0]
                    r = j[1]
                    tmp += np.abs(self.u[u_id].T.dot(self.v[j_id]) - r)
                res += tmp/len(reviews)
            
            print('MAE for test set is {:.4f}'.format( res/len(self.test_user_key)))
            return res/len(self.test_user_key)
        else:
            for i in self.train_user_key.keys():
                tmp = 0
                reviews = self.train_user_key[i]
                for j in reviews:
                    u_id = i
                    j_id = j[0]
                    r = j[1]
                    tmp += np.abs(self.u[u_id].T.dot(self.v[j_id]) - r)
                res += tmp/len(reviews)
            
            print('MAE for training set is {:.4f}'.format( res/len(self.train_user_key)))
            return res/len(self.train_user_key)
    
    def fit(self):
        print('Training started with method', self.method)
        start_time = timeit.default_timer()
        
        if self.method == 'pool':
            
            # set u_i = 1
            for i in self.train_user_key.keys():
                self.u[i] = np.array(1)
                
            # set v_j = mean of all reviews
            for j in self.train_joke_key.keys():
                reviews = self.train_joke_key[j]
                self.v[j] = np.mean([i[1]  for i in reviews ])
                
        elif self.method == 'SVD':
            sparse_matrix = sparse.csr_matrix(self.table)
            
            u,s,vt = svds(sparse_matrix, k=self.d)
            
            for i in range(len(u)):
                self.u[i+1] = u[i]
                
            v = vt.T
            for i in range(len(v)):
                self.v[i+1] = v[i]
        
        elif self.method == 'AM':
            
            for i in range(1, 24984):
                self.u[i] = np.random.randn(self.d)
            
            for j in range(1,101):
                self.v[j] = np.random.randn(self.d)
                
            change = 50
            lamb_matrix = np.diag([self.lamb]*self.d)
            
            iters = 1
            
            while change > 20 and iters < 80:
                change = 0
                
                print('This is iter No.'+str(iters), end='')
                count = 0
                togo = 24984
                interval = togo // 10
                
                
                
                for i in range(1, 24984):
                    reviews = np.array(self.train_user_reviews[i])
                    V = []
                    for j in self.train_user_key[i]:
                        V.append(self.v[j[0]])
                    V = np.array(V)
                    
                    left = V.T.dot(V) + lamb_matrix
                    right = V.T.dot(reviews)
                    new_ui = linalg.solve(left, right)
                    if np.max(np.abs(new_ui - self.u[i])) > 0.1:
                        change += 1
                    
                    self.u[i] = new_ui
                    
                    count += 1
                    if count % interval == 0:
                        print('.', end ='')
                print()
                
                for j in range(1, 101):
                    reviews = np.array(self.train_joke_reviews[j])
                    U = []
                    for i in self.train_joke_key[j]:
                        U.append(self.u[i[0]])
                    U = np.array(U)
                    
                    left = U.T.dot(U) + lamb_matrix
                    right = U.T.dot(reviews)
                    new_vi = linalg.solve(left, right)
                    if np.max(np.abs(new_vi - self.v[j])) > 0.1:
                        change += 5
                    
                    self.v[j] = new_vi
                    
                iters += 1
        
        print('Training completed, time used {:.4f}'.format(
                timeit.default_timer() - start_time))
        
    
    def cv_lamb(self):
        # select lamba
        lamb_pool = [1,2,5,10,15]
        MSE = []
        
        self.train,validation = self.train[0:800000],self.train[800000:]
        
        
        for i in lamb_pool:
            self.lamb = i
            
            self.fit()
            MSE.append(self.MSE('cv', new_data=validation))
            
        plt.plot(lamb_pool, MSE)
        plt.xlabel('lambda')
        plt.ylabel('MSE')
        plt.title('MSE vs lambda using d=10')
        plt.show()
        


