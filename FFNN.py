import json
import math

class InvalidModelException(Exception):
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

if __name__ == "__main__":
    FFNN = FeedForwardNeuralNetwork()
    FFNN.loadModel()
    FFNN.printModel()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = FFNN.predict(inputs)
    container = []
    for i in range(len(inputs)):
        container.append((inputs[i], outputs[i]))
    print("Prediction results:")
    for line in container:
        print(str(line[0]) + " outputs " + str(line[1]))
 