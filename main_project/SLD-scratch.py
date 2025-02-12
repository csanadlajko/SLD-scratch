from PIL import Image
import numpy as np
from glob import glob
import os

def convert(dataset_path, train = True):
    image_size = (100, 100)
    images = []
    labels = []
    data_dict_path = "train"
    if not train:
        data_dict_path = "test"
    
    for label in range(10):
        folder_path = os.path.join(f"{dataset_path}\\{data_dict_path}", str(label))
        images_path = glob(os.path.join(folder_path, "*.jpg"))
        
        for path in images_path:
            image = Image.open(path).convert("L")
            image = image.resize(image_size)
            images.append(np.array(image).flatten())
            labels.append(label)
            
    images = np.array(images) / 255.0
    labels = np.array(labels)
            
    return images, labels

class SLDNeuralNetwork:
    
    """
        The neural network was not able to perform well, so I've decided to refactor the code,
        and re-implement it using batches
    """
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_1_weights = np.random.randn(self.input_nodes, self.hidden_1_nodes) * self.xavier_initialization(self.input_nodes)
        self.hidden_1_to_hidden_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes) * self.xavier_initialization(self.hidden_1_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes) * self.xavier_initialization(self.hidden_2_nodes)
        
        self.hidden_1_biases = np.zeros(self.hidden_1_nodes)
        self.hidden_2_biases = np.zeros(self.hidden_2_nodes)
        self.output_biases = np.zeros(self.output_nodes)
        
    def xavier_initialization(self, prev_nodes):
        return np.sqrt(2. / prev_nodes)
    
    def relu(self, layer_values):
        return np.maximum(0, layer_values)
    
    def relu_derivative(self, layer_values):
        return np.where(layer_values > 0, 1, 0)
    
    def cross_entropy_loss(self, predicted, expected):
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))
    
    def one_hot_encode(self, label):
        result = np.zeros(self.output_nodes)
        result[int(label)] = 1
        return result
    
    def softmax(self, output_values):
        exp_x = np.exp(output_values - np.max(output_values))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def forward(self, input_values):
        self.hidden_1_values = np.dot(input_values, self.input_to_hidden_1_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_hidden_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        
        return self.softmax(self.output_values)
    
    def backward(self, predicted, expected, input_values):
        
        grad_output = predicted - expected
        
        grad_hidden_2 = grad_output.dot(self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        grad_hidden_1 = grad_hidden_2.dot(self.hidden_1_to_hidden_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        
        self.hidden_2_to_output_weights -= self.lr * np.outer(self.hidden_2_values, grad_output)
        self.hidden_1_to_hidden_2_weights -= self.lr * np.outer(self.hidden_1_values, grad_hidden_2)
        self.input_to_hidden_1_weights -= self.lr * np.outer(input_values, grad_hidden_1)
        
        self.output_biases -= self.lr * grad_output
        self.hidden_2_biases -= self.lr * grad_hidden_2
        self.hidden_1_biases -= self.lr * grad_hidden_1

    def train(self, epochs, training_data):
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            np.random.shuffle(training_data)
            for image, label in training_data:
                predicted = self.forward(image)
                expected = self.one_hot_encode(label=label)
                self.backward(predicted=predicted, expected=expected, input_values=image)
                loss_per_epoch += self.cross_entropy_loss(predicted=predicted, expected=expected)
            print(f"Loss at epoch {epoch+1}/{epochs}: {loss_per_epoch/ len(training_data):.4f}")
            
    
    
input_nodes = 100*100
hidden_1_nodes = 128
hidden_2_nodes = 64
output_nodes = 10
learning_rate = 0.01
correct = 0
epochs = 5

images, labels = convert("SLD-scratch\\Sign-Language-Digits-Dataset", train=True)

training_data = list(zip(images, labels))

model = SLDNeuralNetwork(input_nodes=input_nodes, hidden_1_nodes=hidden_1_nodes, hidden_2_nodes=hidden_2_nodes, output_nodes=output_nodes, lr=learning_rate)

model.train(epochs=epochs, training_data=training_data)

test_imgs, test_labels = convert("SLD-scratch\\Sign-Language-Digits-Dataset", train=False)

test_data = list(zip(test_imgs, test_labels))

for image, label in test_data:
    output = model.forward(image)
    formatted = np.argmax(output)
    if formatted == int(label):
        correct += 1

print(f"Neural network accuracy: {100 * correct / len(test_labels):.3f}%")