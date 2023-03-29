# -*- coding: utf-8 -*-
#Function implementation should be done in here
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import csv
import random
random.seed(16)

lambda_0 = 1e-10 #do not use lambda_ = 0 for numerical stability


#_______________________________Functions asked to implement____________________________________________________________________

def compute_error(y, tx, w):
    """
    Computes error
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    
    """
    
    error = y-np.dot(tx,w)
    return error


def compute_sigmoid(x):
    """ 
    Sigmoid function for logistic regression
    Args:
    x: input data
    """
    
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid


def compute_log_pred(tx, w):
    """ 
    Computes the logistic prediction using the w found before
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    
    """
    return (compute_sigmoid(np.dot(tx,w)))
    #activation function for logisitc regression : from a regression method, classifies


def compute_loss(y, tx, w):
    """ 
    Computes the loss with a factor 1/2
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    """
    e = compute_error(y, tx, w)
    loss =  (np.dot(e.T,e)/(2*len(e)))
    return loss


def compute_LL(y, tx, w):
    """ 
    Computes the logarithmic loss: classification loss function
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training    
    """
    pred= tx.dot(w)
    LL = np.mean(np.log(1+np.exp(pred))-y*pred)
    return LL


def compute_grad(y, tx, w):
    """
    Computes the gradient for the gradient descent
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    """
    error = compute_error(y, tx, w)
    grad = (np.dot(tx.T,error))/(-tx.shape[0]) 
    return grad

def compute_log_grad(y, tx, w):
    """
    Computes the gradient for the gradient descent
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    """
    pred = compute_sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def compute_Lgrad(y, tx, w):
    """
    Computes the gradient for the logistic regression
    Args:
    y: output desired values
    tx: input data
    w: weights found in the training
    """
    w = w.ravel()
    y = y.ravel()
    error = y - compute_log_pred(tx, w)
    Lgrad = -tx.T @ error
    Lgrad = np.reshape(Lgrad, (w.shape[0],1))
    return Lgrad 


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    
    Args:
    y: output desired values
    tx: input data
    initial_w: weights found in the training
    max_iters: maximal number of iteration
    gamma: learning rate 
    
    Return:
    w: new weight
    loss: final loss    
    
    """
    w = initial_w
    loss= compute_loss(y,tx,w)
    for n_iter in range(max_iters):
        grad = compute_grad(y, tx, w)
        w = w - gamma*grad
        loss = compute_loss(y, tx, w)
    loss=loss[0][0]
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma): 
    """
    Linear regression using stochastic gradient descent
    
    Args:
    y: output desired values
    tx: input data
    initial_w: weights found in the training
    max_iters: maximal number of iteration
    gamma: learning rate 
    
    Return:
    w: new weight
    loss: final loss
    
    """
    #good_tx = np.nan_to_num(tx)
    w = initial_w
    loss= compute_loss(y,tx,w)
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1):
            grad = compute_grad(batch_y, batch_tx, w)
            w = w - gamma * grad
        loss = compute_loss(y, tx, w)
    loss=loss[0][0]
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    particular case of ridge regression with lambda=0
    
    
    Args:
    y: output desired values
    tx: input data

    
    Return:
    w: new weight
    loss: final loss    
    
    """
    w, loss = ridge_regression(y, tx, lambda_0)
    loss = compute_loss(y, tx, w)
    loss=loss[0][0]
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    
    Args:
    y: output desired values
    tx: input data
    lambda_: L2 regulation
    
    Return:
    w: new weight
    loss: final loss    
    
    """
    N = tx.shape[0]
    D = tx.shape[1]
    tx_new = np.ma.array(tx)
    w = np.linalg.solve(tx.T.dot(tx)+lambda_*2*N*np.eye(D), tx.T.dot(y))
    loss = compute_loss(y, tx, w)  #+ lambda_ * w.T.dot(w).squeeze()
    loss=loss[0][0]
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    Particular case of  regularised logistic regression with lambda_ = 0
    
    
    Args:
    y: output desired values (0,1) values
    tx: input data
    initial_w: weights found in the training
    max_iters: maximal number of iteration
    gamma: learning rate 
    
    Return:
    w: new weight
    loss: final loss    
    
    """
    
    lambda_0 = 1e-10
    w, loss = reg_logistic_regression(y, tx, lambda_0, initial_w, max_iters, gamma)
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularised logistic regression using gradient descent
    
    
    Args:
    y: output desired values (0,1) values
    tx: input data
    initial_w: weights found in the training
    max_iters: maximal number of iteration
    gamma: learning rate 
    
    Return:
    w: new weight
    loss: final loss    
    """
    
    w = initial_w
    loss= compute_LL(y,tx,w)
    for n_iter in range(max_iters):
        #Lgrad = compute_Lgrad(y, tx, w) + 2 * lambda_ * w
        Lgrad = compute_log_grad(y, tx, w) + 2 * lambda_ * w
        w-=gamma * Lgrad
        
    loss = compute_LL(y, tx, w) #+ lambda_ * np.squeeze(w.T.dot(w))
    return w, loss


#_____________________________Functions to process the data_____________________________________________________________________
def separation_validation(X, y, percent=0.8):
    """
    Separates de train data into train and validation set 
    
    Args:
    X: input data
    y: output values
    percent: percentage of the data in the train set 
    
    """
    data_l=X.shape[0]
    choice = np.linspace(0,data_l-1,data_l, dtype=int)
    np.random.shuffle(choice)

    num_train = int(np.floor(data_l * percent))
    num_val = data_l - num_train

    X_train= X[choice[0:num_train],:]
    X_val = X[choice[num_train:],:]
    
    y_train= y[choice[0:num_train]]
    y_val = y[choice[num_train:]]
    return X_train, X_val, y_train, y_val

def k_fold_data(X ,y,k):
    """
    Separates our data in k-fold with train and validation set 
    
    Args:
    X: input data
    y: output values
    k: number of folds
    
    """
    
    #obtenir un vecteur randomisé pour separer en k nos data
    data_l=X.shape[0]
    choice = np.linspace(0,data_l-1,data_l)
    np.random.shuffle(choice)
    choice_sp = np.array(np.split(choice,k))
    choice_sp = choice_sp.astype(int)
    #y.shape = [data_l,1]

    nbr_ligne_fold = int(X.shape[0]/k)
    fold_x = np.zeros((k,nbr_ligne_fold,X.shape[1]))  #initialement nul
    fold_y = np.zeros((nbr_ligne_fold,k)) #initialement nul

    for i in range(k):
        fold_x[i,:,:] = X[choice_sp[i,:],:]
        fold_y[:,i] = y[choice_sp[i,:]]
    return fold_x,fold_y

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Args: 
    y: output desired values
    tx: input data
    
    Return:
    y shuffled -> we will have #num_batches (if =1 -> 1 y)
    tx shuffled (the same way)

    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    
#_______________________Functions to improve the classification_______________________

def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
    tx: input data
    y: output values
    w: weights
        
    Returns:
    The stochastic gradient of the loss at w (same shape as w)
    """

    y_pred = np.dot(tx,w)
    e = y - y_pred
    grad = (np.dot(tx.T,e))/(-tx.shape[0])

    return grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
    tx: input data
    y: output values
    initial_w: weights
    batch_size: scalar denoting the number of data points in a mini-batch 
    max_iters: total number of iterations of SGD
    gamma: learning rate (step size)
        
    Returns:
    losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    

    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y,minibatch_tx,w)
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            w = w - gamma*grad
            #losses.append(loss)
            #ws.append(w)

        #print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w


def logistic_output(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """ 
    Output of logistic regression
    
    Args:
    X: input data
    w: weights of logistic regression
    b: constant, bias
    
    Returns:
    y_hat: Output of logistic regression
    """

    y_hat = compute_sigmoid(X@w+b) #pas la peine de rajouter les 1 dans X #ATTENTION J'AI ENLEVE LE b*np.ones((3,1))
    return y_hat


def bce_loss(X: np.ndarray,  y: np.ndarray, w: np.ndarray, b: float) -> float:
    """ 
    Binary cross-entropy loss function
    
    Args:
    X: Dataset of shape (N, D)
    y: Labels of shape (N, )
    w: Weights of logistic regression model of shape (D, )
    b: bias, a scalar
    
    Returns:
    float: binary cross-entropy loss.
    """
    epsilon = 1e-9
    
    loss = -1/(len(X)) *np.sum(np.dot(y.T, np.log(logistic_output(X,w,b)+epsilon)) +np.dot((1-y).T,np.log(1-logistic_output(X,w,b)+epsilon)))
  
    return loss


def bce_gradient(X: np.ndarray,  y: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """ 
    Gradient of the binary-cross entropy loss
    
    Args:
    X: Dataset of shape (N, D)
    y: Labels of shape (N, )
    w: Weights of logistic regression model of shape (D, )
    b: bias, a scalar
        
    Returns:
    dw (np.ndarray) gradient of the loss with respect to w of shape (D, )
    db (float) gradient of the loss with respect to b, a scalar
    """

    dw = 1/(len(X))* X.T @(logistic_output(X,w,b)-y) # ATTENTION j'ai changé un truc dw = 1/(len(X))* X.T @(logistic_output(X,w,b)-y)
    db = 1/(len(X))*np.sum(logistic_output(X,w,b)-y)
    
    return dw, db


def classify(y_hat: np.ndarray) -> np.ndarray:
    """ 
    Classification function for binary class logistic regression. 
    
    Args:
    y_hat: Output of logistic regression of shape (N, )
    
    Returns:
    labels_pred: Label assignments of data
    """
    labels_pred = np.where(y_hat>0.5, 1,0)
    return labels_pred


def train_logistic_regression(X: np.ndarray, 
                              y: np.ndarray, 
                              max_iters: int = 101, 
                              lr: float = 0.5, 
                              loss_freq: int = 0, 
                              decay : float=1) -> Tuple[np.ndarray, float, dict]:
    
    """ 
    Training function for binary class logistic regression using gradient descent
    
    Args:
    X: Dataset
    y: Labels 
    max_iters: Maximum number of iterations. Default : 100
    lr: The learning rate of  the gradient step. Default : 1
    loss_freq : Prints the loss every `loss_freq` iterations. Default : 0
    decay : Decay of the learning rate. Default : 1
        
    Returns:
    w: weights
    b: scalar, bias
    logger: dict used for visualizations
    """
    
    # Initialize weights
    np.random.seed(0)
    w = np.random.normal(0, 1, size=(X.shape[1], ))
    b = 0

    #Step size for scheduler
    step_size = max_iters//5
    print(f'Step size = {step_size}')
    
    # Initialize dict with lists to keep track of loss, accuracy, weight and bias evolution
    logger = {'loss': [], 
             'acc': [], 
             'w': [],
             'b': []
            }
    
    
    for i in range(max_iters):
        # Compute loss, dw, db and update w and b 
        loss = bce_loss(X,  y, w, b)    #copié-collé les fonctions définies au-dessus
        dw, db = bce_gradient(X,  y, w, b)
        
        w -= lr*dw
        b-= lr*db
        
        # Keep track of parameter, loss and accuracy values for each iteration
        logger['w'].append(w)
        logger['b'].append(b)
        logger['loss'].append(loss)
        y_hat = logistic_output(X, w, b)
        logger['acc'].append(accuracy(y, classify(y_hat)))

        #Scheduler
        if i>11:
            if np.abs(loss-logger["loss"][-10])<0.01:
                lr = lr * decay
        
        if (loss_freq !=0) and i % loss_freq == 0:
            print(f'Loss at iter {i}: {loss:.5f}')
            print(f'Accuracy at iter {i}: ', accuracy(y, classify(y_hat)))
            print(f'Learning rate : {lr:.5f}')
        
    if (loss_freq != 0):
        print('\nFinal loss: {:.5f}'.format(logger['loss'][-1]))
        print('\nFinal accuracy: {:.5f}'.format(logger['acc'][-1]))
        
    return w, b, logger


def accuracy(labels_gt: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Computes the accuracy.

    Args:
    labels_gt: labels (ground-truth) of shape (M, ).
    labels_pred: Predicted labels of shape (M, ).

    Returns:
    float: Accuracy, in range [0, 1].
    """
    correct=0
    for i in range(len(labels_gt)):
        if labels_gt[i]==labels_pred[i]: #être sûre que un à un ils soient égaux
            correct+=1
        
    return correct/len(labels_gt)


def classify_final(y_hat: np.ndarray) -> np.ndarray:
    """ 
    Classification function for binary class logistic regression. 
    
    Args:
    y_hat: Output of logistic regression
    
    Returns:
    Label assignments of data
    """
    labels_pred = np.where(y_hat>0.5, 1,-1)
    return labels_pred


def reg_bce_loss(X: np.ndarray,  y: np.ndarray, w: np.ndarray, b: float, lambda_:float) -> float:
    """ 
    Binary cross-entropy loss function
    
    Args:
    X: Dataset
    y: Labels
    w: Weights of logistic regression model
    b: bias, a scalar
    lambda_ : penalization parameter
    
    Returns:
    float: binary cross-entropy loss.
    """
    # Add the epsilon term to the np.log() in your implementation (e.g. do np.log(x + epsilon) instead of np.log(x))
    # Epsilon is there to avoid log(0)
    epsilon = 1e-9
        
    loss = -1/(len(X)) *np.sum(np.dot(y.T, np.log(logistic_output(X,w,b)+epsilon)) +np.dot((1-y).T,np.log(1-logistic_output(X,w,b)+epsilon))) + lambda_ * np.linalg.norm(w)**2  
    return loss


def reg_bce_gradient(X: np.ndarray,  y: np.ndarray, w: np.ndarray, b: float, lambda_:float) -> Tuple[np.ndarray, float]:
    """ 
    Gradient of the binary-cross entropy loss
    
    Args:
    X: Dataset of shape (N, D)
    y: Labels of shape (N, )
    w: Weights of logistic regression model of shape (D, )
    b: bias, a scalar
    lambda_ : penalization parameter
        
    Returns:
    dw gradient of the loss with respect to w of shape (D, )
    db gradient of the loss with respect to b, a scalar
    """

    dw = 1/(len(X))* X.T @(logistic_output(X,w,b)-y) + lambda_*2*w
    db = 1/(len(X))*np.sum(logistic_output(X,w,b)-y)
    
    return dw, db

def train_reg_logistic_regression(X: np.ndarray, 
                              y: np.ndarray, 
                              max_iters: int = 101, 
                              lr: float = 0.5, 
                              loss_freq: int = 0, 
                              decay : float=1,
                              lambda_:float=0) -> Tuple[np.ndarray, float, dict]:
    """ 
    Training function for binary class logistic regression using gradient descent
    
    Args:
    X: Dataset of shape (N, D)
    y: Labels of shape (N, )
    max_iters: Maximum number of iterations. Default : 100
    lr: The learning rate of  the gradient step. Default : 1
    loss_freq : Prints the loss every `loss_freq` iterations. Default : 0
    decay : Decay of the learning rate. Default : 1
    lambda_ : penalization parameter. Default : 0
        
    Returns:
    w: weights of shape (D, )
    b: scalar
    viz_d: dict used for visualizations
    """
    
    # Initialize weights
    np.random.seed(0)
    w = np.random.normal(0, 1, size=(X.shape[1], ))
    b = 0

    #Step size for scheduler
    step_size = max_iters//5
    print(f'Step size = {step_size}')
    
    # Initialize dict with lists to keep track of loss, accuracy, weight and bias evolution
    logger = {'loss': [], 
             'acc': [], 
             'w': [],
             'b': []
            }
    
    
    for i in range(max_iters):
        # Compute loss, dw, db and update w and b 
        loss = reg_bce_loss(X,  y, w, b, lambda_)    #copié-collé les fonctions définies au-dessus
        dw, db = reg_bce_gradient(X,  y, w, b, lambda_)
        
        w -= lr*dw
        b-= lr*db
        
        # Keep track of parameter, loss and accuracy values for each iteration
        logger['w'].append(w)
        logger['b'].append(b)
        logger['loss'].append(loss)
        y_hat = logistic_output(X, w, b)
        logger['acc'].append(accuracy(y, classify(y_hat)))

        #Scheduler
        if i>11:
            if np.abs(loss-logger["loss"][-1])<0.0001:
                lr = lr * decay
        
        if (loss_freq !=0) and i % loss_freq == 0:
            print(f'Loss at iter {i}: {loss:.5f}')
            print(f'Accuracy at iter {i}: ', accuracy(y, classify(y_hat)))
            print(f'Learning rate : {lr:.5f}')
        
    if (loss_freq != 0):
        print('\nFinal loss: {:.5f}'.format(logger['loss'][-1]))
        print('\nFinal accuracy: {:.5f}'.format(logger['acc'][-1]))
        
    return w, b, logger


def add_constant(X: np.ndarray) -> np.ndarray:
    """ 
    Adds an constant term to the dataset (as the first column)

    Args:
    X (np.ndarray): Dataset of shape (N, D-1)

    Returns: 
    Dataset with offset term added, of shape (N, D)

    """
    X_with_offset = np.insert(X, 0, 1, axis=1)

    return X_with_offset

def indexes_nan(X:np.ndarray, column:int):
    indexes_row=[]
    for i in range(X.shape[0]):
        if X[i,column]!=X[i,column]:
            indexes_row.append(i)
    return indexes_row

def normalize(X, mean, std):
    return((X-mean)/std)

# Plot the evolution of loss during training
def plot_loss(loss_list):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    step = np.arange(1, len(loss_list)+1)
    plt.plot(step, loss_list)
    plt.title('Evolution of the loss during the training')
    plt.xlabel('iteration')
    plt.ylabel('Training loss')
    plt.show()


# Plot the evolution of accuracy during training
def plot_acc(acc_list):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    step = np.arange(1, len(acc_list)+1)
    plt.plot(step, acc_list)
    plt.title('Evolution of the accuracy during the training')
    plt.xlabel('iteration')
    plt.ylabel('Training accuracy')
    
    
def euclidean_dist(sample: np.ndarray, X: np.ndarray):
    """Computes the Euclidean distance between a sample and the training features"""
    distances = np.sqrt(((X - sample) ** 2).sum(axis = 1))
    return distances

def find_nearest_neighbors(
    sample: np.ndarray, 
    X: np.ndarray, 
    k: int = 1):
    """Finds the indices of the k-Nearest Neighbors to a sample
    Args:
        sample: Sample of shape (D, )
        X: Dataset of shape (N, D)
        distance_fn: Distance function
        k: Number of nearest neighbors

    Returns:
        indices: Neighbor indices of shape (k, )
    """
    distances = euclidean_dist(sample, X)
    neighbor_indices = np.argsort(distances)[:k]
    
    return neighbor_indices


def predict_single(
    sample: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray,  
    k: int):
    """ Finds the k-Nearest Neighbors to a sample and returns the majority class
    Args:
        sample: Sample of shape (D, )
        X: Dataset of shape (N, D)
        y: Labels of shape (N, )
        distance_fn: Distance function
        k: number of nearest neighbors

    Returns:
        label: Predicted label, the majority class of the k-Nearest Neighbors

    """
    distances = euclidean_dist(sample, X)
    # From your single sample, get the k closest neighbors among X, using your defined distance function:
    neighbor_indices = find_nearest_neighbors(sample, X,  k)
    # You have the indice list among the dimension data, get their label from the label data:
    neighbor_labels = y[neighbor_indices]

    ideal=np.argmax(neighbor_labels)
   
    return ideal

def predict_1(
    samples: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray, 
    distance_fn, 
    k: int = 1):
    """ Finds the k-Nearest Neighbors to a matrix of samples and returns 
        the majority class for each of these samples as an array
    Args:
        samples: Samples of shape (M, D)
        X: Dataset of shape (N, D)
        y: Labels of shape (N, )
        distance_fn: Distance function
        k: number of nearest neighbors

    Returns:
        labels: Predicted labels of shape (M, )

    """
    predicted_labels = np.apply_along_axis(predict_single, axis=1, arr=samples,
                                           X=X, y=y, distance_fn=euclidean_dist, k=k) 
   
    return predicted_labels



def replace_nans(X, bad_cols) :
    X_train_clean = np.delete(X, bad_cols, axis=1)
    for i in bad_cols:
        nans = indexes_nan(X, i)
        train_x = np.delete(X_train_clean, nans, axis=0)
        train_y = np.delete(X[:,i], nans, axis=0)
        test_x = X_train_clean[nans, :]
        mean = np.mean(train_x, axis=0)
        print(mean)
        std = np.std(train_x, axis=0)
        print(std)
        train_x = normalize(train_x, mean, std)
        test_x = normalize(test_x, mean, std)
        train_x = add_constant(train_x)
        test_x = add_constant(test_x)
        w, loss = least_squares(train_y, train_x)
        test_y = np.dot(test_x, w)
        for j in range(len(nans)):
            X[nans[j], i] = test_y[j]
    return X

##_________________________________________________________________________________________________________________________

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})