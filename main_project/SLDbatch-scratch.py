import numpy as np
from PIL import Image
from glob import glob
import os

def convert(data_path, train=True):
    img_size = (100,100)
    images = []
    labels = []
    data_dict_path = "train"
    if not train:
        data_dict_path = "test"
        
    for label in range(10):
        folder_path = os.path.join(f"{data_path}\\{data_dict_path}", str(label))
        images_path = glob(os.path.join(folder_path, "*.jpg"))
        
        for path in images_path:
            image = Image.open(path).convert("L")
            image = image.resize(img_size)
            images.append(np.array(image).flatten())
            labels.append(label)
    
    images = np.array(images) / 255.0
    labels = np.array(labels)
    
    return images, labels

"""
    TODO -> Documentation, cleaning code
    This file contains a new solution for the SLD neural network
    The implementation uses batches to process the data, giving us a faster and more accurate result
    Also, the first few methods remained the same, but there are some important changes in the backward and training method
"""

class SLDNNBatch():
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_1_weights = np.random.randn(self.input_nodes, self.hidden_1_nodes) * self.he_initialization(self.input_nodes)
        self.hidden_1_to_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes) * self.he_initialization(self.hidden_1_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes) * self.he_initialization(self.hidden_2_nodes)
        
        self.hidden_1_biases = np.zeros(hidden_1_nodes)
        self.hidden_2_biases = np.zeros(hidden_2_nodes)
        self.output_biases = np.zeros(output_nodes)
        
    
    def relu(self, layer_values):
        return np.maximum(0, layer_values)
    
    def relu_derivative(self, layer_values):
        return np.where(layer_values > 0, 1, 0)
    
    def cross_entropy_loss(self, predicted, expected):
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))

    def one_hot_encode(self, batch, batch_labels):
        conv = np.zeros((len(batch), self.output_nodes))
        conv[np.arange(len(batch)), batch_labels] = 1
        return conv
    
    def softmax(self, output_values):
        exp_val = np.exp(output_values - np.max(output_values, axis=1, keepdims=True))
        return exp_val / np.sum(exp_val, axis=1, keepdims=True)
    
    def he_initialization(self, prev_layer):
        return np.sqrt(2. / prev_layer)
    
    
    def forward(self, input_values):
        self.input_values = input_values
        
        self.hidden_1_values = np.dot(self.input_values, self.input_to_hidden_1_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        self.output_values = self.softmax(self.output_values)
        return self.output_values
    
    
    def backward(self, expected):
        
        batch_size = expected.shape[0]
        
        grad_output = self.output_values - expected
        
        grad_hidden_2_to_output_weights = np.dot(self.hidden_2_values.T, grad_output) / batch_size
        grad_output_biases = np.mean(grad_output, axis=0)
        
        grad_hidden_2 = np.dot(grad_output, self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        grad_hidden_1 = np.dot(grad_hidden_2, self.hidden_1_to_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        
        grad_hidden_1_to_2_weights = np.dot(self.hidden_1_values.T, grad_hidden_2) / batch_size
        grad_hidden_2_biases = np.mean(grad_hidden_2, axis=0)
        
        grad_input_to_hidden_weights = np.dot(self.input_values.T, grad_hidden_1) / batch_size
        grad_hidden_1_biases = np.mean(grad_hidden_1, axis=0)
        
        self.hidden_2_to_output_weights -= self.lr * grad_hidden_2_to_output_weights
        self.hidden_1_to_2_weights -= self.lr * grad_hidden_1_to_2_weights
        self.input_to_hidden_1_weights -= self.lr * grad_input_to_hidden_weights
        
        self.output_biases -= self.lr * grad_output_biases
        self.hidden_2_biases -= self.lr * grad_hidden_2_biases
        self.hidden_1_biases -= self.lr * grad_hidden_1_biases
        
    
    def train(self, training_data, epochs, batch_size=16, shuffle=True):
        training_data = np.array(training_data, dtype=object)
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(training_data)
            loss_per_epoch = 0.00
            num_batches = 0
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_imgs = np.array([sample[0] for sample in batch])
                batch_labels = np.array([sample[1] for sample in batch]).astype(int)
                one_hot = self.one_hot_encode(batch=batch, batch_labels=batch_labels)
                
                predicted = self.forward(batch_imgs)
                loss_per_epoch += self.cross_entropy_loss(predicted, one_hot) / len(batch)
                num_batches += 1
                self.backward(one_hot)
            print(f"Loss at epoch {epoch}/{epochs}: {loss_per_epoch / num_batches:.4f}")
            
            
    def test(self, input):
        predicted = self.forward(input)
        return np.argmax(predicted, axis=1)


input_nodes = 100 * 100
hidden_1_nodes = 128
hidden_2_nodes = 64
output_nodes = 10
learning_rate = 0.01
epochs = 200
batch_size = 32

images, labels = convert("SLD-scratch/Sign-Language-Digits-Dataset", train=True)
training_data = list(zip(images, labels))

model = SLDNNBatch(input_nodes=input_nodes,
                         hidden_1_nodes=hidden_1_nodes,
                         hidden_2_nodes=hidden_2_nodes,
                         output_nodes=output_nodes,
                         lr=learning_rate)

model.train(epochs=epochs, training_data=training_data, batch_size=batch_size)

test_imgs, test_labels = convert("SLD-scratch/Sign-Language-Digits-Dataset", train=False)
predictions = model.test(test_imgs)

accuracy = np.mean(predictions == test_labels) * 100
print(f"NN accuracy: {accuracy:.3f}%")