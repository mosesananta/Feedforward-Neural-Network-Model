from gettext import npgettext
import json
import math
import random

class InvalidModelException(Exception):
    def __init__(self):
        super().__init__(self)

class MismatchedArrayLengthException(Exception):
    def __init__(self):
        super().__init__(self)

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.model = None        # Matrix Representation Of Neural Network Layers

    def loadModel(self, file_model="modelSlide1.json"):
        
        with open(file_model) as f:
            self.model = json.load(f)
        try:
            self.validateModel()
        except InvalidModelException:
            print("Error: Invalid Neural Network Model!")
            
        
    def validateModel(self):
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

    def printModel(self):
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
        isValid = self.checkInput(input_matrix)
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
        isValid = self.checkInput(x)
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
            if len(input_list) != len(self.model[0]["neurons"][0])-1:
                print("Error: Invalid Input!!!")
                isValid = False
        return isValid

    def train(self, training_data, mini_batch_size, learning_rate, error_threshold, max_iteration=1000, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        below_threshold = False
        i = 0
        while (i < max_iteration) and not below_threshold:
            total_error = [0 for i in range(len(self.model[-1]["neurons"]))]
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                data_outputs = []
                # Calculate Delta
                mini_batch_error = self.calculate_mini_batch_error(mini_batch, data_outputs)
                # Update Weight
                xs = []
                for data in mini_batch:
                    xs.append(self.predict_data(data["x"])[:-1])
                for data_idx in range(len(mini_batch)):
                    xs[data_idx].insert(0, data[data_idx]["x"])
                self.update_weight(learning_rate, mini_batch_error, xs)
                # Calculate total error
                if self.model[-1]["activation_function"] == "softmax":
                    for data_output in data_outputs:
                        for neuron_idx in range(len(data_output)):
                            total_error[neuron_idx] += -1 * math.log(data_output[neuron_idx])
                else:
                    for data_idx in range(len(mini_batch)):
                        for neuron_idx in range(len(mini_batch[data_idx])):
                            total_error[neuron_idx] += 0.5 * ((mini_batch[data_idx]["y"][neuron_idx] - data_outputs[data_idx][neuron_idx]) ** 2)
            # Check Threshold
            below_threshold = True
            for error_idx in range(len(total_error)):
                if not total_error[error_idx] < error_threshold[error_idx]:
                    below_threshold = False
                    break

    def sum_array(arr1, arr2):
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
            for _ in layer:
                all_deltas[-1].append(0)

        for data in mini_batch:
            # Forward Propagation
            predictions = self.predict_data(data["x"])
            outputs.append(predictions[-1])

            # Calculate Delta
            all_deltas[-1] = self.sum_array(all_deltas[-1], self.calculate_delta_output(predictions[-1], data["y"]))
            for i in range((len(all_deltas[:-1]) - 1), -1, -1):
                all_deltas[i] = self.sum_array(all_deltas[i], self.calculate_delta_hidden(i, predictions[i], all_deltas[i+1]))
        
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
            return self.softmax_derivative(os[neuron_idx], os, ys)
        
    
    def calculate_delta_hidden(self, layer_idx, o_hidden_layer, deltas_from_next_layer, os, ys):
        deltas = []  
        for neuron_idx in range(len(o_hidden_layer)):
            deltas.append(self.calculate_delta_hidden_neuron(layer_idx, neuron_idx, o_hidden_layer[neuron_idx], deltas_from_next_layer, os, ys))
        return deltas

    def calculate_delta_hidden_neuron(self, layer_idx, neuron_idx, o_hidden_neuron, neuron_errors_from_next_layer, os, ys):
        if (self.model[layer_idx]["activation_function"]  == "reLu" or self.model[layer_idx]["activation_function"]  == "sigmoid" or self.model[layer_idx]["activation_function"]  == "linear"):
            if self.model[layer_idx]["activation_function"] == "sigmoid":
                delta = self.sigmoid_derivative(o_hidden_neuron) 
            elif self.model[layer_idx]["activation_function"] == "reLu":
                delta = self.reLu_derivative(o_hidden_neuron)
            else:
                delta = self.linear_derivative(o_hidden_neuron)

            # Sum of Squeres of Errors            
            for i in range(len(neuron_errors_from_next_layer)):
                delta *= neuron_errors_from_next_layer[i] * self.model[layer_idx+1]["neurons"][i][neuron_idx+1]
        else:
            delta = self.softmax_derivative(o_hidden_neuron, os, ys)
        return delta

    def update_weight(self, learning_rate, deltas, xs):
         for layer_idx in range(len(self.model)):
            layer = self.model[layer_idx]
            for neuron_idx in range(len(layer["neurons"])):
                neuron = layer["neurons"][neuron_idx]
                for weight_idx in range(len(neuron)):
                    if weight_idx == 0:
                        neuron[weight_idx] += learning_rate * deltas[layer_idx][neuron_idx]
                    else:
                        x_sum = 0
                        for data in xs:
                            x_sum += data[layer_idx][weight_idx]
                        x_average = x_sum / len(xs)
                        neuron[weight_idx] += learning_rate * deltas[layer_idx][neuron_idx] * x_average

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def linear(self, x):
        return x

    def reLu(self, x):
        return max(0, x)

    def softmax(self, list_of_inputs, neuron, matrix_of_neurons):
        numerator = math.exp(self.dot_product(list_of_inputs, neuron))
        denominator = 0
        for denominator_neuron in matrix_of_neurons:
            denominator += math.exp(self.dot_product(list_of_inputs, denominator_neuron))
        return numerator / denominator
    
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

    def softmax_derivative(self, o , os, ys):
        if os == ys:
            return -1*(1-o)
        else:
            return o

    # Loss functions
    def sum_of_squared_loss(self, output_matrix, y_matrix):
        loss = 0
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[i])):
                loss += 0.5 * (output_matrix[i][j] - y_matrix[i][j]) ** 2
        return loss

   
        

    
        



    
    
    

if __name__ == "__main__":
    FFNN = FeedForwardNeuralNetwork()
    FFNN.loadModel()
    FFNN.printModel()
    inputs = [[0, 0]]
    
    outputs = FFNN.backpropagation(inputs, inputs)

    # container = []
    # for i in range(len(inputs)):
    #     container.append((inputs[i], outputs[i]))
    # print("Prediction results:")
    # for line in container:
    #     print(str(line[0]) + " outputs " + str(line[1]))
 