import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations,data):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.data = data

    keywords=''
    keyLen=0
    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

           

            '''global keywords
            global keyLen
            keyLen=len(self.data)
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

                

                
                nodes = [0, 10, 20, 30, 40]
                sq1dat = [0.799, 0.787, 0.78, 0.776, 0.774]
                #sq1dat = [0.774, 0.776, 0.78, 0.787, 0.799]
                plt.plot(nodes, sq1dat, color='blue')
                plt.xlabel('epoch')
                plt.ylabel('RMSE')
                plt.title('Movielens 10M')
                #plt.legend()
                plt.show()'''    
        
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

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
    
