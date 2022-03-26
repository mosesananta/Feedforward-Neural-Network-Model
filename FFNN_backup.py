from sklearn import datasets
import numpy as np

import json
import math
import random

from sklearn.preprocessing import OneHotEncoder

class InvalidModelException(Exception):
    def __init__(self):
        super().__init__(self)

class MismatchedArrayLengthException(Exception):
    def __init__(self):
        super().__init__(self)

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.model = None        # Matrix Representation Of Neural Network Layers

    def load_model(self, file_model="modelSlide1.json"):
        
        with open(file_model) as f:
            self.model = json.load(f)
        try:
            self.validate_model()
        except InvalidModelException:
            
            print("Error: Invalid Neural Network Model!")

    def init_model(self):
        self.model = []
    
    def create_input_layer(self, number_of_neurons, number_of_inputs, activation_function):
        if self.model is None:
            raise AttributeError("Model not initialized")
        elif len(self.model) != 0:
            raise AttributeError("Model already has an input layer")
        else:
            self.model.append({"neurons": [[random.randint(-1, 1) for _ in range(number_of_inputs + 1)] for _ in range(number_of_neurons)], "activation_function": activation_function})
    
    def append_layer(self, number_of_neurons, activation_function="linear"):
        if self.model is None:
            raise AttributeError("Model not initialized")
        elif len(self.model) < 1:
            raise AttributeError("Model does not have an input layer. Create input layer by calling the method \"create_input_layer\"")
        else:
            self.model.append({"neurons": [[random.randint(-1, 1) for _ in range(len(self.model[-1]['neurons']) + 1)] for _ in range(number_of_neurons)], "activation_function": activation_function})
    
    def export_model(self, filename):
        with open(filename + ".json", "w") as f:
            f.write(json.dumps(self.model, indent = 4))

    def validate_model(self):
        input_count = len(self.model[0]["neurons"][0])
        for neuron in self.model[0]["neurons"]:
            if (len(neuron) != input_count):
                raise InvalidModelException()
        if (self.model[0]["activation_function"] not in ["reLu", "sigmoid", "softmax", "linear"]):
                raise InvalidModelException()

        n_layer = len(self.model[0]["neurons"])
        for layer in self.model[1:]:
            for neuron in layer["neurons"]:
                if (len(neuron) != n_layer + 1): 
                    raise InvalidModelException()
            if (layer["activation_function"] not in ["reLu", "sigmoid", "softmax", "linear"]):
                raise InvalidModelException()
            n_layer = len(layer["neurons"])

    def print_model(self):
        print("Model structure:")
        print("Input layer: " + str(len(self.model[0]["neurons"][0]) - 1) + " features")
        for i in range(len(self.model)):
            if i == (len(self.model) - 1):
                print("Output layer:")
            else:
                print("Hidden layer " + str(i + 1) + ":")
            print("- Activation function: " + str(self.model[i]["activation_function"]))
            for j in range(len(self.model[i]["neurons"])):
                print("- Neuron " + str(j + 1) + ":")
                for k in range(len(self.model[i]["neurons"][j])):
                    print("-- w" + str(k) + " = " + str(self.model[i]["neurons"][j][k]))
         

    def predict(self,input_matrix):
        check_input_matrix = []
        for data in input_matrix:
            check_input_matrix.append({"x": data})
        isValid = self.checkInput(check_input_matrix)
        if not isValid:
            raise SystemExit()
        else:
            # output_matrix = [[0 for i in range(len(input_matrix))] for j in range(len(self.model[0]["neurons"[0]]))]
            calculated_matrix = []
            for input_list in input_matrix:
                calculated_list = input_list     
                for layer in self.model:
                    calculated_list = self.predict_layer(calculated_list, layer["neurons"], layer["activation_function"])
                calculated_matrix.append(calculated_list)
            
            return calculated_matrix

    def predict_data(self, x):
    # Predict result from each neurons
        isValid = True #self.checkInput(x)
        if not isValid:
            raise SystemExit()
        else:
            predictions = []
            predictions.append(self.predict_layer(x, self.model[0]["neurons"], self.model[0]["activation_function"]))
            for layer in self.model[1:]:
                predictions.append(self.predict_layer(predictions[-1], layer["neurons"], layer["activation_function"]))
            return predictions
    
    def predict_layer(self, list_of_inputs, matrix_of_neurons, activation_function):
        list_of_outputs = []
        if activation_function == "linear":
            for neuron in matrix_of_neurons:
                list_of_outputs.append(self.linear(self.dot_product(list_of_inputs, neuron)))
        elif activation_function == "sigmoid":
            for neuron in matrix_of_neurons:
                list_of_outputs.append(self.sigmoid(self.dot_product(list_of_inputs, neuron)))
        elif activation_function == "reLu":
            for neuron in matrix_of_neurons:
                list_of_outputs.append(self.reLu(self.dot_product(list_of_inputs, neuron)))
        else: #softmax
            for neuron in matrix_of_neurons:
                list_of_outputs.append(self.softmax(list_of_inputs, neuron, matrix_of_neurons))
        return list_of_outputs

    def dot_product(self, list_of_inputs, neuron_weights):
        result = neuron_weights[0]

        for i in range(len(list_of_inputs)):
            result += list_of_inputs[i] * neuron_weights[i+1]
        return result

    def checkInput(self,input_matrix):
        isValid = True
        for input_list in input_matrix:
            if len(input_list['x']) != len(self.model[0]["neurons"][0])-1:
                print("Error: Invalid Input!!!")
                isValid = False
        return isValid

    def train(self, training_data, mini_batch_size, learning_rate, error_threshold, max_iteration=1000, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        below_threshold = False
        current_iteration = 0
        while (current_iteration < max_iteration) and not below_threshold:
            total_error = [0 for _ in range(len(self.model[-1]["neurons"]))]
            if self.model[-1]["activation_function"] == "softmax":
                total_error = [0]
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                data_outputs = []
                # Calculate Delta
                mini_batch_error = self.calculate_mini_batch_error(mini_batch, data_outputs)
                for aa in range(len(mini_batch_error)):
                    temp_class = mini_batch_error[aa]
                    for bb in range(len(temp_class)):
                        temp_class[bb] = temp_class[bb] / len(mini_batch)
                # Update Weight
                xs = []
                
                for data in mini_batch:
                    xs.append(self.predict_data(data["x"])[:-1])

                
                for data_idx in range(len(mini_batch)):
                    
                    xs[data_idx].insert(0, mini_batch[data_idx]["x"])

                
                self.update_weight(learning_rate, mini_batch_error, xs)
                # Calculate total error
                if self.model[-1]["activation_function"] == "softmax":
                    for data_idx in range(len(mini_batch)):
                        total_error[0] += self.cross_entropy(mini_batch[data_idx]["y"], data_outputs[data_idx])
                else:
                    for data_idx in range(len(mini_batch)):
                        # print("ys: ", mini_batch[data_idx]["y"])
                        # print("os: ", data_outputs[data_idx])
                        for neuron_idx in range(len(mini_batch[data_idx]["y"])):
                            total_error[neuron_idx] += 0.5 * ((mini_batch[data_idx]["y"][neuron_idx] - data_outputs[data_idx][neuron_idx]) ** 2)
            # Check Threshold
            below_threshold = True
            for error_idx in range(len(total_error)):
                if not (total_error[error_idx] < error_threshold):
                    print("Iteration: " + str(current_iteration) + " Error: " + str(total_error[error_idx]))
                    below_threshold = False
                    break
            current_iteration += 1
            if(below_threshold): print("total error:", total_error)
        return below_threshold, current_iteration



    def cross_entropy(self, y, y_pre):
        if (y_pre[y.index(1)] == 0):
            return (-1) * np.log(0.00001)
        else:
            return (-1) * np.log(y_pre[y.index(1)])
        # y = np.array(y)
        # y_pre = np.array(y_pre)
        # if (0 in y_pre):
        #     print(y_pre)
        # loss = - np.sum(y * np.log(y_pre))
        # return loss / float(y_pre.shape[0]) 
    
    def sum_array(self,arr1, arr2):
        result = []
        arr1_length = len(arr1)
        if (arr1_length != len(arr2)):
            raise MismatchedArrayLengthException()
        else:
            for i in range(arr1_length):
                result.append(arr1[i] + arr2[i])

        return result

    def calculate_mini_batch_error(self, mini_batch, outputs):
        all_deltas = []
        for layer in self.model:
            all_deltas.append([])
            for _ in layer["neurons"]:
                all_deltas[-1].append(0)

        for data in mini_batch:
            # Forward Propagation
            predictions = self.predict_data(data["x"])
            # print("Inputs: ", data["x"])
            # print("Predictions: ", predictions)
            outputs.append(predictions[-1])

            # Calculate Delta
            # # print(all_deltas)
            # # print(all_deltas[-1])
            # # print(self.calculate_delta_output(predictions[-1], data["y"]))
            all_deltas[-1] = self.sum_array(all_deltas[-1], self.calculate_delta_output(predictions[-1], data["y"]))
            for i in range((len(all_deltas[:-1]) - 1), -1, -1):
                all_deltas[i] = self.sum_array(all_deltas[i], self.calculate_delta_hidden(i, predictions[i], all_deltas[i+1],predictions[-1], data["y"]))
        
        return all_deltas

    def calculate_delta_output(self, os, ys):
        deltas = []
        for neuron_idx in range(len(self.model[-1]["neurons"])):
            deltas.append(self.calculate_delta_output_neuron(neuron_idx, os, ys))
        return deltas

    def calculate_delta_output_neuron(self, neuron_idx, os, ys):
        if self.model[-1]["activation_function"] == "sigmoid":
            return (ys[neuron_idx] - os[neuron_idx]) * self.sigmoid_derivative(os[neuron_idx])
        elif self.model[-1]["activation_function"] == "linear":
            return (ys[neuron_idx] - os[neuron_idx]) * self.linear_derivative(os[neuron_idx])
        elif self.model[-1]["activation_function"] == "reLu":
            return (ys[neuron_idx] - os[neuron_idx]) * self.reLu_derivative(os[neuron_idx])
        else:
            return self.softmax_derivative(neuron_idx, os, ys)
        
    
    def calculate_delta_hidden(self, layer_idx, o_hidden_layer, deltas_from_next_layer, os, ys):
        deltas = []  
        for neuron_idx in range(len(o_hidden_layer)):
            deltas.append(self.calculate_delta_hidden_neuron(layer_idx, neuron_idx, o_hidden_layer[neuron_idx], deltas_from_next_layer, os, ys))
        return deltas

    def calculate_delta_hidden_neuron(self, layer_idx, neuron_idx, o_hidden_neuron, neuron_errors_from_next_layer, os, ys):
        if (self.model[layer_idx]["activation_function"]  == "reLu" or self.model[layer_idx]["activation_function"]  == "sigmoid" or self.model[layer_idx]["activation_function"]  == "linear"):
            sigma_term = 0
            for i in range(len(neuron_errors_from_next_layer)):
                sigma_term += neuron_errors_from_next_layer[i] * self.model[layer_idx+1]["neurons"][i][neuron_idx+1]
            if self.model[layer_idx]["activation_function"] == "sigmoid":
                delta = sigma_term * self.sigmoid_derivative(o_hidden_neuron) 
            elif self.model[layer_idx]["activation_function"] == "reLu":
                delta = sigma_term * self.reLu_derivative(o_hidden_neuron)
            else:
                delta = sigma_term * self.linear_derivative(o_hidden_neuron)
        else:
            delta = self.softmax_derivative(neuron_idx, os, ys)
        return delta

    def update_weight(self, learning_rate, deltas, xs):
        for layer_idx in range(len(self.model)):
            layer = self.model[layer_idx]
            for neuron_idx in range(len(layer["neurons"])):
                neuron = layer["neurons"][neuron_idx]

                for weight_idx in range(len(neuron)):
                    if weight_idx == 0:
                        neuron[weight_idx] += (-1) * learning_rate * deltas[layer_idx][neuron_idx]
                    else:
                        x_sum = 0
                        for data in xs:
                            x_sum += data[layer_idx][weight_idx - 1]
                       
                        x_average = x_sum / len(xs)
                        # print("x_average: ", x_average)
                       
                        neuron[weight_idx] += (-1) * learning_rate * deltas[layer_idx][neuron_idx] * x_average
                        

    # Activation functions
    def sigmoid(self, x):
        # if (x > 1000):
        #     print("x: ", x)
        try:
            res = 1 / (1 + math.exp(-round(x, 5)))
            return res
        except OverflowError:
            return 0

    def linear(self, x):
        return x

    def reLu(self, x):
        return max(0, x)

    def softmax(self, list_of_inputs, neuron, matrix_of_neurons):
        # print("--------------NEURON---------------------")
        # print(neuron)
        # print("--------------Matrix of Neuron---------------------")
        # print(matrix_of_neurons)
        # print("--------------End of Matrix of Neuron---------------------")
        try: 
            numerator = math.exp(self.dot_product(list_of_inputs, neuron))
            
            denominator = 0
            for denominator_neuron in matrix_of_neurons:
                denominator += math.exp(self.dot_product(list_of_inputs, denominator_neuron))
            return numerator / denominator
            
        # if (denominator == 0):
        except OverflowError:
            return 0
    
    # Activation functions derivatives
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def linear_derivative(self, x):
        return 1

    def reLu_derivative(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def softmax_derivative(self, neuron_idx, os, ys):
        if ((ys[neuron_idx] > 0.5) and (os[neuron_idx] > 0.5)) or ((ys[neuron_idx] <= 0.5) and (os[neuron_idx] <= 0.5)):
            return -1 * (1 - os[neuron_idx])
        else:
            return os[neuron_idx]

    # Loss functions
    def sum_of_squared_loss(self, output_matrix, y_matrix):
        loss = 0
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[i])):
                loss += 0.5 * (output_matrix[i][j] - y_matrix[i][j]) ** 2
        return loss
    
    def onehot_encode(list_of_output):
        encoded_output = {}
        output_label = []
        for output in list_of_output:
            if output not in output_label:
                output_label.append(output)
        for output in list_of_output:
            list_of_zero = [0 for _ in range(len(output_label))]
            list_of_zero[output_label.index(output)] = 1
            encoded_output[output_label.index(output)] = list_of_zero
        return encoded_output
    
  


            
 
    

if __name__ == "__main__":

    iris = datasets.load_iris()

    iris_list = []
    encoded_labels = FeedForwardNeuralNetwork.onehot_encode(iris.target)
    for i in range(len(iris.data)):
        temp = {}
        temp['x'] = iris.data[i]
        temp['y'] = encoded_labels[iris.target[i]]
        iris_list.append(temp)

    FFNN = FeedForwardNeuralNetwork()
    # FFNN.load_model("iris_last2.json")
    FFNN.init_model()
    FFNN.create_input_layer(2, 4, "sigmoid")
    FFNN.append_layer(10, "sigmoid")
    FFNN.append_layer(1, "sigmoid")
    FFNN.append_layer(3, "softmax")
    FFNN.print_model()



    Convergence,iteration = FFNN.train(iris_list,10,0.001,0.01,1000)

    print("Is Convergen?",Convergence,"Iteration:",iteration)
    FFNN.print_model()

    FFNN.export_model("iris_last_test")

# Testing
    # test = iris.data[50:60]
    # print(test)
    # test_target = iris.target[100:110]

    # outputs = FFNN.predict(test)

    # accuracy = 0
    # print(test_target)
    # print("---------")
    # print(outputs)
    # print("-----------")
    # print("benar", accuracy,"/10")

    # inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # outputs = FFNN.predict(inputs)
    # container = []
    # for i in range(len(inputs)):
    #     container.append((inputs[i], outputs[i]))
    # print("Prediction results:")
    # for line in container:
    #     print(str(line[0]) + " outputs " + str(line[1]))

    # container = []
    # for i in range(len(inputs)):
    #     container.append((inputs[i], outputs[i]))
    # print("Prediction results:")
    # for line in container:
    #     print(str(line[0]) + " outputs " + str(line[1]))
 