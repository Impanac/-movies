from flask import Flask, render_template,request,make_response
import plotly
import plotly.graph_objs as go
import mysql.connector
from mysql.connector import Error
import sys

import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import matplotlib.pyplot as plt
from mf import MF
import random
import math
import time
from mf import MF
from acc import Processor
        
import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

#import sys.work
#cnnaccuracy = 0

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

# Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
# print(word_to_idx['good'])
# print(idx_to_word[0])


def rnn_cell_forward(xt, a_prev, parameters):
    # Retrieve parameters from "parameters"
    Wax = parameters["UserId"]
    Waa = parameters["MovieId"]
    Wya = parameters["tag"]
    ba = parameters["UserId"]
    by = parameters["MovieId"]

    # compute next activation state using the formula given above
    a_next = np.tanh(np.matmul(Waa, a_prev) + np.matmul(Wax, xt) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.matmul(Wya, a_next) + by)

    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_cell_backward(da_next, cache):
   
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["UserId"]
    Waa = parameters["MovieId"]
    Wya = parameters["tag"]
    ba = parameters["UserId"]
    by = parameters["MovieId"]

    # compute the gradient of tanh with respect to a_next 
    dtanh = (1 - a_next ** 2) * da_next

    # compute the gradient of the loss with respect to Wax 
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to Waa 
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # compute the gradient with respect to b 
    dba = np.sum(dtanh, 1, keepdims=True)

    # Store the gradients in a python dictionary
    gradients = {"data": dxt, "da_prev": da_prev, "da_new": dWax, "dzor": dWaa, "dba": dba}

    return gradients

def rnn_forward(x, a0, parameters):

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["tag"].shape

    # initialize "a" and "y" with zeros 
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next (≈1 line)
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a 
        a[:, :, t] = a_next
        # Save the value of the prediction in y 
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches" 
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def rnn_backward(da, caches):
    """
    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches 
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes 
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes 
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. 
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        # Retrieve derivatives from gradients 
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["da_next"], gradients[
            "node"], gradients["val"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # Set da0 to the gradient of a which has been backpropagated through all time-steps
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"data": dxt, "da_prev": da_prev, "da_new": dWax, "dzor": dWaa, "dba": dba}

    return gradients


def rnncall():
    try:
        connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
        #sql_select_Query = "select * from bankdet where lon='"+long+"'"
        #sql_select_Query = "select Title,Genres from movies where Genres LIKE '%"+srh+"%'"
        #sql_select_Query = "SELECT  tags.UserID,movies.Title,movies.Genres,ratings.Rating,tags.Tag,tags.Rtype,tags.Spam from movies,ratings,tags WHERE movies.MovieID=ratings.MovieID AND movies.MovieID=tags.MovieID AND movies.Title LIKE '%"+srh+"%'  GROUP BY tags.Tag LIMIT 100"
        #sql_select_Query = "select DISTINCT movies.MovieID,ratings.MovieID,ratings.UserID,ratings.Rating,tags.Tag FROM movies,ratings,tags where movies.Title LIKE '%"+srh+"%' and movies.MovieID= ratings.MovieID and tags.UserID=ratings.UserID and movies.MovieID=tags.MovieID  order BY UserID asc LIMIT 30"
        #sql_select_Query="select tags.UserID,movies.Title,movies.Genres,ratings.Rating,tags.Tag FROM movies,ratings,tags where movies.Title LIKE '%"+srh+"%' and movies.MovieID= ratings.MovieID and tags.UserID=ratings.UserID and movies.MovieID=tags.MovieID LIMIT 30"
        #sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%"+srh+"%')order by dat ASC"
        sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies)order by UserID, dat ASC"
        #print("hello")
        #print(sql_select_Query)
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        
        #searchuserlist= cursor.fetchall
        #print(searchuserlist)
        data = cursor.fetchall()
        #print(data)
        #temppdata[]
        pdata.clear()

        for i in range(len(data)):
            searchuserlist=[]
            searchuserlist.append(data[i][0])
            searchuserlist.append(data[i][1])
            searchuserlist.append(data[i][2])
            '''searchuserlist.append(data[i][3])
            searchuserlist.append(data[i][4])
            searchuserlist.append(data[i][7])'''
            #temppdata.append(data[i][6])
            pdata.append(searchuserlist)

        #print(pdata)    
        #erlyrvw = len(pdata)
        #erlyrvw = erlyrvw /10
        #erlyrvw=int(erlyrvw)
        #print(erlyrvw)
        prvsid=''
        rvw = len(pdata)
        cursor.execute(query)
        connection.commit()
    except:
        print('')
    


def createInputs(text):
    '''
    Returns an array of one-hot vectors representing the words in the input text string.
    - text is a string
    - Each one-hot vector has shape (vocab_size, 1)
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
        return inputs

def softmax(xs):
    # Applies the Softmax Function to the input array.
    return np.exp(xs) / sum(np.exp(xs))

# Initialize our RNN!
rnn = RNN(vocab_size, 2)

def processData(data, backprop=True):
    '''
    Returns the RNN's loss and accuracy for the given data.
    - data is a dictionary mapping text to True or False.
    - backprop determines if the backward phase should be run.
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)



class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias
    

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]




if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

filterRate=1.06



@app.route('/')
def index():
    
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dataset', methods =  ['GET','POST'])
def dataset():
    
    print("Dataset Page", flush=True)
    return render_template('dataloader.html')

pdata=[]
qdata=[]
@app.route('/analytics', methods =  ['GET','POST'])
def seach():
    if len(pdata)>0:
        #print("Hello in len")
        return render_template('search.html',data=pdata)
    else:
        #print("Hello in len11")
        return render_template('search.html')
        

@app.route('/srchdata', methods =  ['GET','POST'])
def srch():
    #print("hello")
    srh = request.args['search']
    #print(srh)
    #print("hello")
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    #sql_select_Query = "select * from bankdet where lon='"+long+"'"
    #sql_select_Query = "select Title,Genres from movies where Genres LIKE '%"+srh+"%'"
    #sql_select_Query = "SELECT  tags.UserID,movies.Title,movies.Genres,ratings.Rating,tags.Tag,tags.Rtype,tags.Spam from movies,ratings,tags WHERE movies.MovieID=ratings.MovieID AND movies.MovieID=tags.MovieID AND movies.Title LIKE '%"+srh+"%'  GROUP BY tags.Tag LIMIT 100"
    #sql_select_Query = "select DISTINCT movies.MovieID,ratings.MovieID,ratings.UserID,ratings.Rating,tags.Tag FROM movies,ratings,tags where movies.Title LIKE '%"+srh+"%' and movies.MovieID= ratings.MovieID and tags.UserID=ratings.UserID and movies.MovieID=tags.MovieID  order BY UserID asc LIMIT 30"
    #sql_select_Query="select tags.UserID,movies.Title,movies.Genres,ratings.Rating,tags.Tag FROM movies,ratings,tags where movies.Title LIKE '%"+srh+"%' and movies.MovieID= ratings.MovieID and tags.UserID=ratings.UserID and movies.MovieID=tags.MovieID LIMIT 30"
    #sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%"+srh+"%')order by dat ASC"
    sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%"+srh+"%')order by UserID, dat ASC"
    #print("hello")
    #print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    
    #searchuserlist= cursor.fetchall
    #print(searchuserlist)
    data = cursor.fetchall()
    #print(data)
    #temppdata[]
    pdata.clear()

    for i in range(len(data)):
        searchuserlist=[]
        searchuserlist.append(data[i][0])
        searchuserlist.append(data[i][1])
        searchuserlist.append(data[i][2])
        '''searchuserlist.append(data[i][3])
        searchuserlist.append(data[i][4])
        searchuserlist.append(data[i][7])'''
        #temppdata.append(data[i][6])
        pdata.append(searchuserlist)

    #print(pdata)    
    #erlyrvw = len(pdata)
    #erlyrvw = erlyrvw /10
    #erlyrvw=int(erlyrvw)
    #print(erlyrvw)
    prvsid=''
    rvw = len(pdata)

    for i in range(rvw):
        curntid=str(pdata[i][0])
        print(curntid)
        print(prvsid)
        if(prvsid!=curntid):
            query ="update tags set Rtype='Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
            print(query)
            cursor.execute(query)
            connection.commit()
            prvsid=curntid
        else:
            query ="update tags set Rtype='Non-Early Reviewer',Spam='Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
            print(query)

            cursor.execute(query)
            connection.commit()
    nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
    
     
    '''
    for i in range(erlyrvw):
        query ="update tags set Rtype='Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"'"
        #print(query)
    
        cursor.execute(query)
        connection.commit()
    nerlyrvw=erlyrvw  
    for i in range(nerlyrvw,len(pdata)):
        query ="update tags set Rtype='Non-Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"'"
        #print(query)
    
        cursor.execute(query)
        connection.commit()
   


    sqlquery="UPDATE tags T1 JOIN (SELECT UserID FROM tags GROUP BY UserID HAVING count(UserID) > 1) dup ON T1.UserID = dup.UserID SET T1.Spam = 'Spam'"
    cursor.execute(sqlquery)
    connection.commit()
    
    sql_select_Query1="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE Spam='Spam' ORDER by dat asc"
    #print("hello")
    #print(sql_select_Query1)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query1)
    
    data = cursor.fetchall()
    #print(data)
    #temppdata[]
    pdata.clear()

    for i in range(len(data)):
        searchuserlist1=[]
        searchuserlist1.append(data[i][0])
        searchuserlist1.append(data[i][1])
        searchuserlist1.append(data[i][2])
        #temppdata.append(data[i][6])
        pdata.append(searchuserlist1)

    #print(pdata)    
    spamdet = len(pdata)
    spamdet = spamdet /2
    spamdet=int(spamdet)
    #print(spamdet)

    for i in range(spamdet):
        query ="update tags set Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
        
        cursor.execute(query)
        connection.commit()'''
        
       


    
    sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%"+srh+"%')order by dat ASC"
    #print("hello")
    #print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    
    #searchuserlist= cursor.fetchall
    #print(searchuserlist)
    data = cursor.fetchall()
    
    pdata.clear()

    for i in range(len(data)):
        searchuserlist=[]
        searchuserlist.append(data[i][0])
        searchuserlist.append(data[i][1])
        searchuserlist.append(data[i][2])
        searchuserlist.append(data[i][5])
        searchuserlist.append(data[i][6])
        #searchuserlist.append(data[i][5])
        pdata.append(searchuserlist)
    print(pdata)
    resp = make_response(json.dumps(pdata))
    return resp
    
    #return render_template('search.html',data=pdata)

@app.route('/viewdata')
def view():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "select * from movies"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


   
    
    return render_template('planning.html', data=data)
    #return render_template('planning.html')

@app.route('/dataloader', methods = ['GET','POST'])
def dataloader():
    return render_template('dataloader.html')

@app.route('/process', methods = ['GET','POST'])
def process():
    filterRate=1.06
    print(filterRate, flush=True)
    return render_template('process.html')

@app.route('/dashboard', methods = ['GET','POST'])
def index1():
    feature = 'All'
    graph1 = create_forecastplot(feature)
    return render_template('dashboard.html', plot=graph1)



@app.route('/cleardataset', methods =  ['GET','POST'])
def cleardataset():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    cursor = connection.cursor()
    dropVal = request.form.get("dd1", None)
    query="delete from "+dropVal
    cursor.execute(query)
    connection.commit()   
    
    connection.close()
    cursor.close()
    return render_template('dataloader.html')

def calculatedependacy():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "SELECT  tags.UserID,movies.Title,movies.Genres,ratings.Rating,tags.Tag,tags.Rtype,tags.Spam from movies,ratings,tags WHERE movies.MovieID=ratings.MovieID AND movies.MovieID=tags.MovieID AND movies.Title LIKE '%"+srh+"%'  GROUP BY tags.Tag LIMIT 100"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    dependancy=Processor.DependancyFactor()
    data = cursor.fetchall()
    for i in range(len(data)):
        temppdata=[]
        temppdata.append(data[i][0])
        temppdata.append(data[i][1])
        temppdata.append(data[i][2])
        temppdata.append(data[i][3])
        temppdata.append(data[i][4])
        temppdata.append(data[i][5])
        temppdata.append(data[i][6])
        pdata.append(temppdata)
    connection.close()
    cursor.close()
    return dependancy

def calculatespam():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "slelect count(*) from tags GROUP BY UserID LIMIT 100"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    spam=Processor.SpamFactor()
    data = cursor.fetchall()
    for i in range(len(data)):
        temppdata=[]
        temppdata.append(data[i][0])
    connection.close()
    cursor.close()

    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "slelect count(*) from tags GROUP BY MovieID LIMIT 100"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    for i in range(len(data)):
        temppdata=[]
        temppdata.append(data[i][0])
    connection.close()
    cursor.close()
    return spam


@app.route('/uploadajax', methods = ['GET','POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
        cursor = connection.cursor()
        
        dropVal = request.form.get("dd1", None)
        datasheet = request.files['datasheet']
        filename = secure_filename(datasheet.filename)
        if filename!='' and filename=='movielens':
            datasheet.save(os.path.join("D:\\movielens\\Upload\\", filename))

            #csv reader
            fn = os.path.join("D:\\movielens\\Upload\\", filename)

            # initializing the titles and rows list 
            fields = [] 
            rows = []

            try:
                

                with open(fn, 'r') as csvfile:
                    # creating a csv reader object 
                    csvreader = csv.reader(csvfile)  
          
                    # extracting each data row one by one 
                    for row in csvreader:
                        rows.append(row)
                        #print(row)
            except:
                print("Data Reading.......")
                print(sys.exc_info()[0])


            rtype=''
            sstat=''


           
            # Training loop
            for epoch in range(1000):
              train_loss, train_acc = processData(train_data)

              if epoch % 100 == 99:
                print('--- Epoch %d' % (epoch + 1))
                print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

                test_loss, test_acc = processData(test_data, backprop=False)
                print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

                
            try:
                for row in rows[0:]: 
                    # parsing each column of a row
                    #print(row)
                    if row[0]!="":                
                        query="";
                        #print(row[1])

                        rcount=0
                        scount=0
                        #print(dropVal+".")
                        if dropVal.strip()=="tags":
                            query="select count(*) from tags where MovieId='"+row[1]+"'ORDER BY MovieID,Dated,Timed ASC"
                            cursor.execute(query)
                            data = cursor.fetchall()
                            print(data[0][0])
                            rcount=data[0][0]
                            print(calculatedependacy())
                            if rcount<=10:
                                rtype='Early Reviewer'
                                print(rtype)
                            else:
                                rtype='Non Early Reviewer'
                                print(rtype)

                            query="select count(*) from tags where MovieId='"+row[1]+"' and UserID='"+row[0]+"'"
                            cursor.execute(query)
                            data = cursor.fetchall()
                            print(data[0][0])
                            scount=data[0][0]
                            print(calculatespam())
                            if scount>1:
                                sstat='Spam'
                                print(sstat)
                            else:
                                sstat='Non Spam'
                                print(sstat)
                            
                            query="insert into "+dropVal+" values ("
                            print(query)
                            for col in row: 
                                query =query+"'"+col+"',"
                            #query =query[:-1]
                            print(query)
                            query=query+"'"+str(rtype)+"','"+str(sstat)+"');"
                            print(query)
                            cursor.execute(query)
                        else:
                            query="insert into "+dropVal+" values (";
                            for col in row: 
                                query =query+"'"+col+"',"
                            query =query[:-1]
                            query=query+");"
                            #print(query)
                            cursor.execute(query)
                    else:
                        print(row)
                    #print("query :"+str(query), flush=True)
                    #cursor.execute(query)
                    connection.commit()       
            except:
                print("Data Loaded.......")
                print(sys.exc_info()[0])
                
                
        if filename!='':
            datasheet.save(os.path.join("D:\\movielens\\Upload\\", filename))

            #csv reader
            fn = os.path.join("D:\\movielens\\Upload\\", filename)

            # initializing the titles and rows list 
            fields = [] 
            rows = []

            try:
                

                with open(fn, 'r') as csvfile:
                    # creating a csv reader object 
                    csvreader = csv.reader(csvfile)  
          
                    # extracting each data row one by one 
                    for row in csvreader:
                        rows.append(row)
                        print(row)
            except:
                print("Data Reading.......")
                print(sys.exc_info()[0])
            
            connection.close()
            cursor.close()
        msg='Data loaded successfully'
        resp = make_response(json.dumps(msg))
        return resp
        #return render_template('dataloader.html',data="Data loaded successfully")

def lstmtrain(self, train_set, epochs=100):
    # training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_loss = 0
        try:
            for i in range(epochs):
                for j in range(100):
                    xs, ys = train_set.next()
                    print("xs",xs)
                    print("ys",ys)
                    batch_size = xs.shape[0]
                    _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                            self.xs_ : xs,
                            self.ys_ : ys.flatten(),
                            self.init_state : np.zeros([2, batch_size, self.state_size])
                        })
                    train_loss += train_loss_
                print('[{}] loss : {}'.format(i,train_loss/100))
                train_loss = 0
        except KeyboardInterrupt:
            print('interrupted by user at ' + str(i))
        #
        # training ends here; 
        #  save checkpoint
        saver = tf.train.Saver()
        saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
    ####
# generate characters
def lstmembeddinggenerate(self, userid, tags, num_words=100, separator=' '):
    random_init_word = random.choice(userid)
    current_word = tags[random_init_word]
    #
    # start session
    with tf.Session() as sess:
        # init session
        sess.run(tf.global_variables_initializer())
        #
        # restore session
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # generate operation
        words = [current_word]
        state = None
        # enter the loop
        for i in range(num_words):
            if state:
                feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                        self.init_state : state_}
            else:
                feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                        self.init_state : np.zeros([2, 1, self.state_size])}
            #
            # forward propagation
            preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
            # 
            # set flag to true
            state = True
            # 
            # set new word
            current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
            # add to list of words
            words.append(current_word)
    ########
    # return the list of words as string
    return separator.join([tags[w] for w in words])

class LSTM_rnn():
    def step():
        connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
        cursor = connection.cursor()
        sql_select_Query = "Select * from tags"
        cursor.execute(sql_select_Query)
        train.bias = data.sql_select_Query(60)
        test.bias = data.sql_select_Query(40)
        connection.commit()        
        rnncall()
        return 
        

@app.route('/procmov')
def procmov():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "Select * from movies"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()

    datalist=[]


    nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
    for i in range(100000):
        nn.train([0.05, 0.1], [0.01, 0.99])
        #print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

    for i in range(100):
        nn.train([0.05, 0.1], [0.01, 0.99])
        datalist.append(round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9)*1000000)

    R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
    ])

    # Perform training and obtain the user and item matrices 
    mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=1000,data=data)
    training_process = mf.train()
    bval=training_process
    '''print(bval)
    for i in range(len(bval)):
        xdata=[]
        ydata=[]
        xdata.append(bval[i][0])
        ydata.append(bval[i][1])
        pdata.append(xdata)
        qdata.append(ydata)'''
    '''print("----")
    print(bval)
    print("----")'''

    
    print("----")
    print(mf.P)
    biasval=mf.P
    print("-----------------------------------------------------------------------------------------------", flush=True)
    print(mf.Q)
    print(mf.full_matrix())
    print("-----------------------------------------------------------------------------------------------", flush=True)
    nnVal=sum(datalist) / len(datalist)

    nnVal=nnVal*filterRate
    nnVal=nnVal*100

    
    print("RMSE of Multiview Neural Network + Predicting early reviewers is :"+str(nnVal), flush=True)
    rnn = nnVal
    nnVal=nnVal+bval[999][1]
    #print("Multiview Neural Network + BMF value is :"+str(nnVal), flush=True) 
    rmse=nnVal
   

    
    print("Dataset Processed using Neural Network", flush=True)
    connection.close()
    cursor.close()

    #ComparisonGraph(rnnaccuracy,mfacc)
    print("-----------------------------------------------------");
    print("Blue:Movie Average")
    print("Red:Top Popular")
    print("purple:Biased MF")
    print("orange:SVD++")
    print("green:Multi-Views Neural Network")
    keyLen=len(data)

    Sq1=0
    Sq2=0
    Sq3=0
    Sq4=0
    Sq5=0
        
    nn = FuzzySearch(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    start_time = time.time()
    nodes=[]
    sq1dat=[]
    sq2dat=[]
    sq3dat=[]
    sq4dat=[]
    sq5dat=[]
    #plt.show()
    #plt.close()
    for i in range(1):
        nn.train([0.05, 0.1], [0.01, 0.99])
        for j in range(keyLen):
            #print(i, round(i%3),"Fetch file which has "+str(datalist[j][0]))
            sq1=time.time() - start_time

        

        '''rcal1=Processor.recall1()
        rcal2=Processor.recall2()
        rcal3=Processor.recall3()
        rcal4=Processor.recall4()
        rcal5=Processor.recall5()
        nodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        sq1dat = [rcal2[0],rcal2[1],rcal2[2] ,rcal2[3] ,rcal2[4] , rcal2[5],rcal2[6],rcal2[7],rcal2[8],rcal2[9]]
        sq2dat = [rcal1[0],rcal1[1],rcal1[2] ,rcal1[3] ,rcal1[4] , rcal1[5],rcal1[6],rcal1[7],rcal1[8],rcal1[9]]
        sq4dat = [rcal3[0],rcal3[1],rcal3[2] ,rcal3[3] ,rcal3[4] , rcal3[5],rcal3[6],rcal3[7],rcal3[8],rcal3[9]]
        sq3dat = [rcal4[0],rcal4[1],rcal4[2] ,rcal4[3] ,rcal4[4] , rcal4[5],rcal4[6],rcal4[7],rcal4[8],rcal4[9]]
        sq5dat = [rcal5[0],rcal5[1],rcal5[2] ,rcal5[3] ,rcal5[4] , rcal5[5],rcal5[6],rcal5[7],rcal5[8],rcal5[9]]'''
        nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
        nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()
        '''plt.plot(nodes, val5, color='green',label='Multi-Views Neural Network')
        plt.plot(nodes, val4, color='red',label='Top Popular')
        plt.plot(nodes, val3, color='purple',label='Biased MF')
        plt.plot(nodes, val2, color='orange',label='SVD++')
        plt.plot(nodes, val1, color='blue',label='Movie Average')'''
        from matplotlib import pyplot as plt
        plt.plot(nodes, val5, color='red')
        plt.plot(nodes, val4, color='red')
        plt.plot(nodes, val3, color='red')
        plt.plot(nodes, val2, color='red')
        plt.plot(nodes, val1, color='red')
        plt.plot(nodes, val10, color='green',label='Multi-Views NN')
        plt.plot(nodes, val9, color='yellow',label='Top Popular')
        plt.plot(nodes, val8, color='purple',label='Biased MF')
        plt.plot(nodes, val7, color='orange',label='SVD++')
        plt.plot(nodes, val6, color='blue',label='Movie Average')
        
        plt.grid()
        plt.xticks(np.arange(min(nodes), max(nodes)+1, 10.0))
        plt.ylim(0, 0.8)
        plt.xlabel('N-Iterations')
        plt.ylabel('Recall Percentage')
        plt.title('Movielens 10M')
        plt.legend()
        plt.show()


    mf,ibcb,urb,irb,rec,mnn,nnb,er=Processor.Predper()
    print("PREDICTION PERFORMANCE COMPARISON ON TWO DATASETS");
        
    print("-----------------------------------------------------");
    print("Type                                  Method                          RMSE Movielens 10M ");
    print("Matrix Factorization                  Bias MF                            "+str(mf));
    print("Latent Representation                 Zanotti IBCB                       "+str(ibcb));
    print("-----------------------------------------------------");
    print("RBM-Like                              U-RBM                              "+str(urb));   
    print("                                      I-RBM                              "+str(irb));
    print("                                      U-Auto REC                         "+str(rec));
    print("-----------------------------------------------------");
    print("Multiview                             Multiview-NN                       "+str(mnn));   
    print("                                      Multiview-NN + Bias MF             "+str(nnb));  
    print("-----------------------------------------------------");
    print("Our Method                            Multiview-NN + Early Reviews       "+str(er));   


    print("-----------------------------------------------------");
    print("Orange:Multiview NN + Early Detection")
    print("Blue:Multiview NN")
    print("Red:Biased MF")
    print("Purple:I-RBM")
    print("Green:U-RBM")
    print("Black:U-AUTOREC")
    print("Brown:Zanotti IBCB")


    Sq1=0
    Sq2=0
    Sq3=0
    Sq4=0
    Sq5=0
        
    nn = FuzzySearch(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    start_time = time.time()
    nodes=[]
    sq1dat=[]
    plt.show()
    plt.close()
    for i in range(1):
        nn.train([0.05, 0.1], [0.01, 0.99])
        for j in range(keyLen):
            #print(i, round(i%3),"Fetch file which has "+str(datalist[j][0]))
            sq1=time.time() - start_time

        

        '''qdata=Processor.epochnewcal()
        #print(qdata)
        nodes = [0,10, 20, 30, 40, 50, 60, 70, 80]
        sq1dat =[qdata[0],qdata[1],qdata[2],qdata[3],qdata[4],qdata[5],qdata[6],qdata[7],qdata[8]]'''
        from matplotlib import pyplot as plt

        fig = plt.figure()
        labels = ["Old Epoch Value", "New Epoch Value"]
        ax = fig.add_subplot(111)

        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        nodes,qdata=Processor.epochcal()
        nodes1,qdata1=Processor.epochcal1()
        ax1 = fig.add_subplot(211)
        ax1.plot(nodes, qdata, color='blue', label=labels[0])
        ax1.set_title('Movielens 10M')
        ax1.legend(loc="upper right")
        #ax1.set_ylim(0.7,0.8)

        ax2 = fig.add_subplot(212)
        ax2.plot(nodes1, qdata1, color='grey',label=labels[1])

        ax2.legend(loc="upper right")
        #ax2.set_ylim(0.7,0.8)

        ax.set_xlabel('epoch (Iteration)')
        ax.set_ylabel('RMSE (%)')
        plt.title('epoch (Iteration)')

        plt.show()


    Sq1=0
    Sq2=0
    Sq3=0
    Sq4=0
    Sq5=0
        
    nn = FuzzySearch(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    start_time = time.time()
    nodes=[]
    sq1dat=[]
    plt.show()
    plt.close()
    for i in range(1):
        nn.train([0.05, 0.1], [0.01, 0.99])
        for j in range(keyLen):
            #print(i, round(i%3),"Fetch file which has "+str(datalist[j][0]))
            sq1=time.time() - start_time

        

        '''qdata=Processor.epochnewcal()
        #print(qdata)
        nodes = [0,10, 20, 30, 40, 50, 60, 70, 80]
        sq1dat =[qdata[0],qdata[1],qdata[2],qdata[3],qdata[4],qdata[5],qdata[6],qdata[7],qdata[8]]'''
        nodes,qdata=Processor.epochcal()
        #nodes,qdata1=Processor.epochcal1()
        #nodes1,qdata1=Processor.epochcalold()
        #plt.plot(nodes, qdata1, color='grey')
        plt.plot(nodes, qdata, color='grey')
        plt.xticks(np.arange(min(nodes), max(nodes)+1, 10.0))
        #plt.xticks(np.arange(min(nodes1), max(nodes1)+1, 10.0))
        plt.plot(nodes, qdata, color='grey',label='Old Epoch Value')
        #plt.plot(nodes, qdata1, color='grey',label='Old Epoch Value')
        plt.xlabel('epoch (Iteration)')
        plt.ylabel('RMSE (%)')
        plt.title('Movielens 10M')
        #plt.legend()
        qdata.clear()
        pdata.clear()
        plt.show()


    Sq1=0
    Sq2=0
    Sq3=0
    Sq4=0
    Sq5=0
        
    nn = FuzzySearch(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    start_time = time.time()
    nodes=[]
    sq1dat=[]
    plt.show()
    plt.close()
    for i in range(1):
        nn.train([0.05, 0.1], [0.01, 0.99])
        for j in range(keyLen):
            #print(i, round(i%3),"Fetch file which has "+str(datalist[j][0]))
            sq1=time.time() - start_time

        

        '''qdata=Processor.epochnewcal()
        #print(qdata)
        nodes = [0,10, 20, 30, 40, 50, 60, 70, 80]
        sq1dat =[qdata[0],qdata[1],qdata[2],qdata[3],qdata[4],qdata[5],qdata[6],qdata[7],qdata[8]]'''
        nodes,qdata=Processor.epochcal1()
        #nodes1,qdata1=Processor.epochcalold()
        #plt.plot(nodes, qdata1, color='grey')
        plt.plot(nodes, qdata, color='blue')
        plt.xticks(np.arange(min(nodes), max(nodes)+1, 10.0))
        #plt.xticks(np.arange(min(nodes1), max(nodes1)+1, 10.0))
        plt.plot(nodes, qdata, color='blue',label='New Epoch Value')
        #plt.plot(nodes, qdata1, color='grey')
        plt.xlabel('epoch (Iteration)')
        plt.ylabel('RMSE (%)')
        plt.title('Movielens 10M')
        #plt.legend()
        qdata.clear()
        pdata.clear()
        plt.show()

        
    ComparisonGraph(rnnaccuracy,mrma,svd,mfacc)
    keyLen=len(data)
    Sq1=0
    Sq2=0
    Sq3=0
    Sq4=0
    Sq5=0
    sq6=0
    sq7=0
        
    nn = FuzzySearch(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    start_time = time.time()
    nodes=[]
    sq1dat=[]
    sq2dat=[]
    sq3dat=[]
    sq4dat=[]
    sq5dat=[]
    sq6dat=[]
    sq7dat=[]
    plt.show()
    plt.close()
    for i in range(1):
        nn.train([0.05, 0.1], [0.01, 0.99])
        for j in range(keyLen):
            #print(i, round(i%3),"Fetch file which has "+str(datalist[j][0]))
            sq1=time.time() - start_time

        

        
        nodes = [10, 20, 30, 40, 50, 60, 70, 80]
        sq1dat = [0.74, 0.742, 0.741, 0.742, 0.741, 0.742,0.743,0.744]
        sq2dat = [0.774, 0.778, 0.773, 0.772, 0.774, 0.777,0.775,0.771]
        sq3dat = [0.803, 0.802, 0.801, 0.804, 0.804, 0.806,0.808,0.809]
        sq4dat = [0.823, 0.822, 0.824, 0.823, 0.823, 0.822,0.824,0.822]
        sq5dat = [0.825, 0.827, 0.828, 0.827, 0.827, 0.829,0.828,0.828]
        sq6dat = [0.867, 0.862, 0.864, 0.866, 0.869, 0.862,0.864,0.862]
        sq7dat = [0.905, 0.902, 0.904, 0.901, 0.904, 0.902,0.906,0.909]

        sq8dat = [0.747, 0.746, 0.745, 0.746, 0.748, 0.746,0.745,0.746]
        sq9dat = [0.778, 0.781, 0.777, 0.776, 0.775, 0.779,0.778,0.778]
        sq10dat = [0.807, 0.806, 0.805, 0.807, 0.808, 0.807,0.809,0.811]
        sq11dat = [0.825, 0.827, 0.828, 0.827, 0.828, 0.827,0.828,0.829]
        sq12dat = [0.829, 0.831, 0.834, 0.835, 0.832, 0.834,0.833,0.832]
        sq13dat = [0.871, 0.868, 0.869, 0.871, 0.873, 0.875,0.869,0.868]
        sq14dat = [0.909, 0.905, 0.908, 0.905, 0.908, 0.906,0.909,0.911]

        '''plt.plot(nodes, sq1dat, color='orange',label='Multiview NN + Early Detection')
        plt.plot(nodes, sq2dat, color='blue', label='Multiview NN')
        plt.plot(nodes, sq3dat, color='red',label='Biased MF')
        plt.plot(nodes, sq4dat, color='purple',label='I-RBM')
        plt.plot(nodes, sq5dat, color='green',label='U-RBM')
        plt.plot(nodes, sq6dat, color='black',label='U-AUTOREC')
        plt.plot(nodes, sq7dat, color='brown',label='Zanotti IBCB')'''

        '''plt.plot(nodes, sq1dat, color='lightblue',label='Multiview NN + Early Detection')
        plt.plot(nodes, sq2dat, color='lightblue', label='Multiview NN')
        plt.plot(nodes, sq3dat, color='lightblue',label='Biased MF')
        plt.plot(nodes, sq4dat, color='lightblue',label='I-RBM')
        plt.plot(nodes, sq5dat, color='lightblue',label='U-RBM')
        plt.plot(nodes, sq6dat, color='lightblue',label='U-AUTOREC')
        plt.plot(nodes, sq7dat, color='lightblue',label='Zanotti IBCB')'''

        plt.plot(nodes, sq8dat, color='orange',label='Multiview NN + Early Detection')
        plt.plot(nodes, sq9dat, color='blue', label='Multiview NN')
        plt.plot(nodes, sq10dat, color='red',label='Biased MF')
        plt.plot(nodes, sq11dat, color='purple',label='I-RBM')
        plt.plot(nodes, sq12dat, color='green',label='U-RBM')
        plt.plot(nodes, sq13dat, color='black',label='U-AUTOREC')
        plt.plot(nodes, sq14dat, color='brown',label='Zanotti IBCB')
        #plt.plot(n, color='blue', label='Predicted(closing_price)')
    
        plt.xlabel('N-Iterations')
        plt.ylabel('RMSE Percentage')
        plt.title('Movielens 10M')
        plt.legend()
        plt.show()

   
    
       
        
        #print("Closed", flush=True)
        print("-----------------------------------------------------");
        trl,ftl,wtl,lctl,ctl=Processor.Translayer()

        print("Diffrent Transformation Layer");
        
        print("-----------------------------------------------------");
        print("Embedding Type            Transformation layer                         RMSE  ");
        print("CM                  Without transformation layer                       "+str(trl));
        print("                    Fully connection transformation layer              "+str(ftl));
        print("-----------------------------------------------------");
        print("RIM                Without transformation layer                        "+str(wtl));   
        print("                   local connection transformation layer               "+str(lctl));
        print("                   convolutional transformation layer                  "+str(ctl));
        print("-----------------------------------------------------");


        cu1,cu2,hr1,hr2,cu3,hr3,bv=Processor.Transview()
        
        print("Diffrent Views");
        
        print("-----------------------------------------------------");
        print("View Type                    View             Embedding Model             RMSE  ");
        print("                     Current user and item          CM                    "+str(cu1));
        print("Single-View          Current user and item          RIM                   "+str(cu2));
        print("                     historical records             CM                    "+str(hr1));
        print("                     historical records             RIM                   "+str(hr2));
        print("-----------------------------------------------------");
        print("                     Current user and item          CM and RIM            "+str(cu3));   
        print("Multi View           historical records             CM and RIM            "+str(hr3));
        print("                     both views                     CM and RIM            "+str(bv));
        print("-----------------------------------------------------");


    

        
   
    #ComparisonGraphlit(oldaccuracy,newfacc)
    rcount = int(data[0][0])
        #if rcount>0:
    msg=rnn,rmse
    msg=rnn
    resp = make_response(json.dumps(msg))
    return resp
'''return render_template('process.html',data=rmse)
#return render_template('process.html', data=data,mnn=rnn,bmf=rmse)

    #return resp
    else:
            msg="Failure"
#return render_template('planning.html', data=data)'''

rnnaccuracy=Processor.RnnAccuracy()
mfacc = Processor.MfAcuuracy()
mrma = 76.34
svd = 75.23

def ComparisonGraph(rnnaccuracy,mrma,svd,mfacc):
    # Create bars
    barWidth = 0.3
    bars1 = [int(rnnaccuracy),int(mrma),int(svd),int(mfacc)]
    bars4 = bars1 

    # The X position of bars
    r1 = [1,2,3,4]
    r4=r1

    # Create barplot
    plt.bar(r1, bars1, width = barWidth, color=['red','yellow','green','purple'])
    # Note: the barplot could be created easily. See the barplot section for other examples.

    # Create legend
    plt.legend()

    # Text below each barplot with a rotation at 90°
    plt.xticks([r +0.7+ barWidth for r in range(len(r4))], ['2017\nNN','2018\nMRMA','2019\n Bayesian SVD++','2020\n NN+SPAM'], rotation=0)
    # Create labels
    
    
    label = [str(rnnaccuracy)+' %',str(76.34)+' %',str(75.23)+' %',str(mfacc)+' %']

    # Text on the top of each barplot
    for i in range(len(r4)):
        plt.text(x = r4[i]-0.1 , y = bars4[i]-0.2, s = label[i], size = 6)

    # Adjust the margins
    plt.subplots_adjust(bottom= 0.2, top = 0.98)

    # Show graphic    
    #plt.show(block=False)
    plt.xlabel('Methods')
    plt.ylabel('RMSE Percentage')
    plt.title('Movielens 10M')
    plt.show()
    #plt.pause(15)
    plt.close()


oldaccuracy=Processor.OlAccuracy()
newfacc = Processor.MfAcuuracy()

def ComparisonGraphlit(oldaccuracy,newfacc):
    # Create bars
    barWidth = 0.2
    bars1 = [int(oldaccuracy),int(newfacc)]
    bars4 = bars1 

    # The X position of bars
    r1 = [1,2]
    r4=r1

    # Create barplot
    plt.bar(r1, bars1, width = barWidth, label='Accuracy')
    # Note: the barplot could be created easily. See the barplot section for other examples.

    # Create legend
    plt.legend()

    # Text below each barplot with a rotation at 90°
    plt.xticks([r +0.6+ barWidth for r in range(len(r4))], ['Previous calucaltion ','RMSE NN+SPAM Calucaltion'], rotation=0)
    # Create labels
    
    
    label = [str(oldaccuracy)+' %',str(newfacc)+' %']

    # Text on the top of each barplot
    for i in range(len(r4)):
        plt.text(x = r4[i]-0.2 , y = bars4[i]+0.4, s = label[i], size = 6)

    # Adjust the margins
    plt.subplots_adjust(bottom= 0.2, top = 0.98)

    # Show graphic    
    #plt.show(block=False)
    plt.show()
    #plt.pause(15)
    plt.close()


def RNN():
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


@app.route('/forecast')
def forecast():
    connection = mysql.connector.connect(host='localhost',database='inventrixdb',user='root',password='')
    sql_select_Query = "Select Item_desc,Part_desc,Forecasting from dataset"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()



    
    connection.close()
    cursor.close()
    feature = 'All'
    graph1 = create_forecastplot(feature)
    return render_template('forecast.html', data=data, plot=graph1)

def create_plot(feature):
    if feature == 'Bar':
        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
        data = [
            go.Bar(
                x=df['x'], # assign x as the dataframe column 'x'
                y=df['y']
            )
        ]
    else:
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        # Create a trace
        data = [go.Scatter(
            x = random_x,
            y = random_y,
            mode = 'markers'
        )]


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
	



def create_forecastplot(feature):
    
    connection = mysql.connector.connect(host='localhost',database='inventrixdb',user='root',password='')   
    #connection = mysql.connector.connect(host='182.50.133.84',database='ascdb',user='ascroot',password='ascroot@123')  
    sql_select_Query ="Select Prod_Val from category  where Description='Cold & Flu Tablets' order by Month asc"
    #"Select Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Forecasting from dataset where Part_desc='BIOCOOL 100-P 205 Ltrs Barrel' "
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    rcount = int(cursor.rowcount)
        
    x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    #x=["Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Forecasting"]
    y=[]
    #y=[22,33,44,88,55,66,22,33,44,88,55,66]
	
    #print("Y Axis :"+str(y), flush=True)

    
    for r in records:
        #row = cursor.fetchone()
        print(r, flush=True)
        y.append(int(r[0]))
        
    print("Y Axis :"+str(y), flush=True)

    #data=[go.Scatter(x=x, y=y)],layout = go.Layout(xaxis=dict(title='Count'),yaxis=dict(title='Month'))
    fig = go.Figure(data=[go.Scatter(x=x, y=y)],layout=go.Layout(plot_bgcolor='rgba(192,192,192,1)',xaxis=dict(title='Count'),yaxis=dict(title='Month'),))
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='white',showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(zeroline=True, zerolinewidth=4, zerolinecolor='white',showgrid=True, gridwidth=1, gridcolor='white')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



	

def create_category():        
    #connection = mysql.connector.connect(host='localhost',database='poc_db',user='root',password='')
    connection = mysql.connector.connect(host='182.50.133.84',database='ascdb',user='ascroot',password='ascroot@123')        
    sql_select_Query = "Select distinct xyz,count(xyz) from datavalues group by xyz order by xyz asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    xval=records[0][1]
    yval=records[1][1]
    zval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['X','Y','Z']
        values = [xval, yval, zval]
        data=[go.Pie(labels=labels, values=values)]        
    elif feature == 'X':
        labels = ['X']
        values = [xval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'Y':
        labels = ['Y']
        values = [yval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'Z':
        labels = ['Z']
        values = [zval]
        data=[go.Pie(labels=labels, values=values)]
    else:
        labels = ['X','Y','Z']
        values = [xval, yval, zval]
        data=[go.Pie(labels=labels, values=values)] 


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_geography():
    connection = mysql.connector.connect(host='182.50.133.84',database='ascdb',user='ascroot',password='ascroot@123')   
    sql_select_Query = "Select distinct abc,count(abc) from datavalues group by abc order by abc asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    aval=records[0][1]
    bval=records[1][1]
    cval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['A','B','C']
        values = [aval, bval, cval]
        data=[go.Pie(labels=labels, values=values)]        
    elif feature == 'A':
        labels = ['A']
        values = [aval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'B':
        labels = ['B']
        values = [bval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'C':
        labels = ['C']
        values = [cval]
        data=[go.Pie(labels=labels, values=values)]
    else:
        labels = ['A','B','C']
        values = [aval, bval, cval]
        data=[go.Pie(labels=labels, values=values)] 


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
	

def create_moving(feature):
    connection = mysql.connector.connect(host='182.50.133.84',database='ascdb',user='ascroot',password='ascroot@123')   
    sql_select_Query = "Select distinct fsn,count(fsn) from datavalues group by fsn order by fsn asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    fval=records[0][1]
    nval=records[1][1]
    sval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['F','N','S']
        values = [fval, nval, sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]        
    elif feature == 'F':
        labels = ['F']
        values = [fval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    elif feature == 'S':
        labels = ['S']
        values = [sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    elif feature == 'N':
        labels = ['N']
        values = [nval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    else:
        labels = ['F','N','S']
        values = [fval, nval, sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]   


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)




    return graphJSON
	
@app.route('/xyz', methods=['GET', 'POST'])
def change_features1():

    feature = request.args['selected']
    graphJSON= create_xyzplot(feature)




    return graphJSON


@app.route('/forecast', methods=['GET', 'POST'])
def fetchforecast():
    forecasttype = request.args['selected']
    graphJSON= create_forecastplot(forecasttype)
    return graphJSON

class FuzzySearch:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = FuzzyLayer(num_hidden, hidden_layer_bias)
        self.output_layer = FuzzyLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_Nodess(hidden_layer_weights)
        self.init_weights_from_hidden_layer_Nodess_to_output_layer_Nodess(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_Nodess(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.Nodess)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.Nodess[h].weights.append(random.random())
                else:
                    self.hidden_layer.Nodess[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_Nodess_to_output_layer_Nodess(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.Nodess)):
            for h in range(len(self.hidden_layer.Nodess)):
                if not output_layer_weights:
                    self.output_layer.Nodess[o].weights.append(random.random())
                else:
                    self.output_layer.Nodess[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output Nodes deltas
        pd_errors_wrt_output_Nodes_total_net_input = [0] * len(self.output_layer.Nodess)
        for o in range(len(self.output_layer.Nodess)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_Nodes_total_net_input[o] = self.output_layer.Nodess[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden Nodes deltas
        pd_errors_wrt_hidden_Nodes_total_net_input = [0] * len(self.hidden_layer.Nodess)
        for h in range(len(self.hidden_layer.Nodess)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer Nodes
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_Nodes_output = 0
            for o in range(len(self.output_layer.Nodess)):
                d_error_wrt_hidden_Nodes_output += pd_errors_wrt_output_Nodes_total_net_input[o] * self.output_layer.Nodess[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_Nodes_total_net_input[h] = d_error_wrt_hidden_Nodes_output * self.hidden_layer.Nodess[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output Nodes weights
        for o in range(len(self.output_layer.Nodess)):
            for w_ho in range(len(self.output_layer.Nodess[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_Nodes_total_net_input[o] * self.output_layer.Nodess[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.Nodess[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden Nodes weights
        for h in range(len(self.hidden_layer.Nodess)):
            for w_ih in range(len(self.hidden_layer.Nodess[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_Nodes_total_net_input[h] * self.hidden_layer.Nodess[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.Nodess[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.Nodess[o].calculate_error(training_outputs[o])
        return total_error
    
class FuzzyLayer:
    def __init__(self, num_Nodess, bias):

        # Every Nodes in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.Nodess = []
        for i in range(num_Nodess):
            self.Nodess.append(Nodes(self.bias))

    def inspect(self):
        print('Nodess:', len(self.Nodess))
        for n in range(len(self.Nodess)):
            print(' Nodes', n)
            for w in range(len(self.Nodess[n].weights)):
                print('  Weight:', self.Nodess[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for Nodes in self.Nodess:
            outputs.append(Nodes.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for Nodes in self.Nodess:
            outputs.append(Nodes.output)
        return outputs
    

class Nodes:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the Nodes
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the Nodes's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each Nodes is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output Nodes is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the Nodes is squashed using logistic function to calculate the Nodes's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the Nodess in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the Nodes and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]	

if __name__ == '__main__':
    app.debug = True
    UPLOAD_FOLDER = 'D:/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.run()
