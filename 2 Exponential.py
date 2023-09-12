import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#################################   
#################################    training data

d0 = pd.read_csv('./mnist_train.csv')
l_train = np.array(d0['label'])
d_train = np.array(d0.drop("label",axis = 1))
del d0
zeros_train = []
eights_train = []

def display(index,vector):
    plt.figure(figsize=(7,7))
    grid_data = vector[index].reshape(28,28)
    plt.imshow(grid_data,interpolation = "none",cmap = "Greys")
    plt.show()
    
i = 0
while(i<60000):
    if ((l_train[i] == 0)and(len(zeros_train)<5500)):
        zeros_train.append(d_train[i])
    elif((l_train[i] == 8)and(len(eights_train)<5500)):
        eights_train.append(d_train[i])
    i = i + 1

del l_train
del d_train
zeros_train = np.array(zeros_train)    # 5923x784
eights_train = np.array(eights_train)  # 5852x784

zeros_train = zeros_train.astype('float32')
zeros_train /= 255
eights_train = eights_train.astype('float32')
eights_train /= 255

#################################   
#################################    NETWORK 2

def ReLU(X):           
    return np.maximum(0,X)

def der_ReLU(X):
    return np.where(X>0,[1],[0])

def linear(X):
    return X

def delta_An(X,Bn,h,k):
    return (np.multiply(Bn,der_ReLU(h)))*(X.T)

def delta_an(X,Bn,h,k):
    return (np.multiply(Bn,der_ReLU(h)))

def delta_Bn(X,Bn,h,k):
    return h

def delta_bn(X,Bn,h,k):
    return 1

def delta_J_to_matrix(y0,y1,f):
    return (  math.exp(0.5*y0)*f(t_input_0,Bn,h,k) - math.exp((-0.5)*y1)*f(t_input_1,Bn,h,k) )

def J_net2(y0,y1):
    return math.exp(0.5*y0) + math.exp((-0.5)*y1) 


An = np.array([[np.random.normal(0,(1/22)) for i in range(784)] for j in range(300)]) #300x784
an = np.array([[0 for j in range(300)]]).T  #300x1
Bn = np.array([[np.random.normal(0,(1/21))] for j in range(300)])    #300x1  #μ=0.5
bn = 0

μ = 10**(-4)
i=0
j=0
iterations = 0 
Jav_new = 0
Jav_old = 0
epochs = 0
Jlist = []
Jlist20 = []  # holds batches of 20 of cost function to take average from

#train network 2

while(True):

    t_input_0 = (np.array([zeros_train[i]])).T  #784x1
    z = np.dot(An,t_input_0) + an    #300x1
    h = ReLU(z)
    k = (np.dot(Bn.T,h) + bn)[0][0]           #scalar
    y0 = linear(k)

    t_input_1 = (np.array([eights_train[i]])).T  #784x1
    z = np.dot(An,t_input_1) + an   #20x1
    h = ReLU(z)
    k = (np.dot(Bn.T,h) + bn)[0][0]           #scalar
    y1 = linear(k)

    Ann = An - μ*delta_J_to_matrix(y0,y1,delta_An)  # SGD
    ann = an - μ*delta_J_to_matrix(y0,y1,delta_an)
    Bnn = Bn - μ*delta_J_to_matrix(y0,y1,delta_Bn)
    bnn = bn - μ*delta_J_to_matrix(y0,y1,delta_bn)

    An = Ann
    an = ann
    Bn = Bnn
    bn = bnn

    Jlist20.append(J_net2(y0,y1))  # holds 20 costs to take average

    if(j==19):              # convergence test
        Jav_new = sum(Jlist20)/len(Jlist20)
        Jlist.append(Jav_new)
        if( abs(Jav_new - Jav_old) <= 10**(-6) ):   #order of accuracy         
            break
        Jav_old = Jav_new
        Jlist20 = []
        j = 0
    
    i = i + 1
    if(i == 5500):       # input again the same training data but shuffled
        i = 0
        np.random.shuffle(zeros_train)  #shuffles rows 
        np.random.shuffle(eights_train)
        epochs = epochs + 1
        
    j = j + 1
    iterations = iterations + 1
    
    if(epochs == 5):
        break

print("Finished training network 2")
print("Iterations to train network 2 = %d" %iterations)
print("Epochs to train network 2 = %d" %epochs)

plt.xlabel('Every 20nth iteration')
plt.ylabel('Cost in log scale')
plt.title('Convergence of Network 2')
plt.plot(Jlist)
plt.yscale('log')
plt.show()

# end of training network 2

An_opt_net2 = Ann  
an_opt_net2 = ann
Bn_opt_net2 = Bnn
bn_opt_net2 = bnn
θ_opt_net2 = [An_opt_net2,an_opt_net2,Bn_opt_net2,bn_opt_net2] #θο of network 2

#################################   
#################################    testing data

d0 = pd.read_csv('./mnist_test.csv')
l_test = np.array(d0['label'])
d_test = np.array(d0.drop("label",axis = 1))
del d0
zeros_test = []
eights_test = []

i = 0
while(i<10000):
    if (l_test[i] == 0):
        zeros_test.append(d_test[i])
    elif(l_test[i] == 8):
        eights_test.append(d_test[i])
    i = i + 1

del l_test
del d_test
zeros_test = np.array(zeros_test)  #holds all numbers 0 for testing
eights_test = np.array(eights_test) #holds all numbers 8 for testing

zeros_test = zeros_test.astype('float32')  # 980x784
zeros_test /= 255
eights_test = eights_test.astype('float32') # 974x784
eights_test /= 255

#test network 2

i = 0
s01 = 0  # times that x = [x1,x2,...,x784] comes from H0 but H1 is decided
s10 = 0  # times that x = [x1,x2,...x784] comes from H1 but H0 is decided
while(i<973):      
    
    x_input_0 = (np.array([zeros_test[i]])).T  #forwarding input coming from H0
    z = np.dot(An_opt_net2,x_input_0) + an_opt_net2
    h = ReLU(z)
    k = ( np.dot(Bn_opt_net2.T,h) + bn_opt_net2 )[0][0]       
    y0 = linear(k)  # output of network 1 for input x coming from HO 

    if(y0>=0):
        s01 = s01 + 1

    x_input_1 = (np.array([eights_test[i]])).T #forwarding input coming from H1
    z = np.dot(An_opt_net2,x_input_1) + an_opt_net2     
    h = ReLU(z)
    k = ( np.dot(Bn_opt_net2.T,h) + bn_opt_net2 )[0][0]        
    y1 = linear(k)  # output of network 1 for input x coming from H1 

    if(y1<0):
        s10 = s10 + 1

    i = i + 1

# end of testing network 2

p_error_net2_H0 = s01 / 974   
p_error_net2_H1 = s10 / 974

print("Ποσοστό σφάλματος για την υπόθεση H0 του δικτύου 1 (exponential) = %f" % p_error_net2_H0)
print("Ποσοστό σφάλματος για την υπόθεση H1 του δικτύου 1 (exponential) = %f" % p_error_net2_H1)
print("Αθροιστικό ποσοστό σφάλματος = %f" % ((p_error_net2_H0 + p_error_net2_H1)/2))


