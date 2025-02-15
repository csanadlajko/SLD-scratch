import numpy as np
from PIL import Image
from glob import glob
import os

"""
    TODO -> The current ANN performs quite poorly, even after changing the hyperparameters several times.
    The next step is to implement a CNN solution in order to get much optimal results.
"""

def convert(data_path, train=True):
    """
        The convert method's main input is SLD dictionary's path, which contains the training and testing data.
        Throughout the function, we convert the images into numpy arrays and normalize them.
        The function gives the images (np.ndarray) and labels (int stored in np.ndarray) as outputs.
    """
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


class SLDNNBatch():
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        """
            We initialize the the numbers of layers, and number of artificial neurons that will the layers contain.
            We also declare the learning rate, giving it 0.01 as default value.
            Additionally, we initialize the weights between layers and the biases for the hidden and the output layers.
        """
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
        """
            Activation function, returns 0 if the layer's value is negative,
            else the output is the layer value itself.
        """
        return np.maximum(0, layer_values)
    
    def relu_derivative(self, layer_values):
        """
            Derivative of the ReLU activation function.
            Returns 1 if the layer's value is positive, else the output is 0.
        """
        return np.where(layer_values > 0, 1, 0)
    
    def cross_entropy_loss(self, predicted, expected):
        """
            The function returns the current output's loss, based on the predicted and expected output.
        """
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))

    def one_hot_encode(self, batch, batch_labels):
        """
            This function returns the expected vectors as numpy arrays.
            As we work in batches in this solution, we return a numpy array corresponding to that size.
        """
        conv = np.zeros((len(batch), self.output_nodes))
        conv[np.arange(len(batch)), batch_labels] = 1
        return conv
    
    def softmax(self, output_values):
        """
            Converts the output values into probabilities, 
            so it'll be easier to determine the neural networks choice using the argmax function.
        """
        exp_val = np.exp(output_values - np.max(output_values, axis=1, keepdims=True))
        return exp_val / np.sum(exp_val, axis=1, keepdims=True)
    
    def he_initialization(self, prev_layer):
        """
            A method used for modifying the weights in order to get better results.
        """
        return np.sqrt(2. / prev_layer)
    
    
    def forward(self, input_values):
        """
            The forward method passes and modifies the input data,
            sending them through the weights to the output layer.
            The function returns the probabilities for the current number.
        """
        
        self.input_values = input_values
        
        self.hidden_1_values = np.dot(self.input_values, self.input_to_hidden_1_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        self.output_values = self.softmax(self.output_values)
        return self.output_values
    
    
    def backward(self, expected):
        """
            The backward method is responsible for the backpropagation.
            As we work in batches, we calculate the gradient values corresponding to that.
            We use np.mean while updating the bias values, because as mentioned above, we are updating batches of data. 
        """
        
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
        
    
    def train(self, training_data, epochs, batch_size=64, shuffle=True):
        """
            The train method trains the dataset based on the given training data and epochs.
            As for the batch_size arguement, I gave 64, but feel free to change it anytime in order to improve the neural network's accuracy.
            Throughout the training loop we use the forward and backward methods, implemented above
        """
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
        """
            Returns the predicted class labels fot given inputs.
        """
        predicted = self.forward(input)
        return np.argmax(predicted, axis=1)


"""
    Inititializing the hyperparameters and training the model.
    This model returns a decent accuracy if the number of epochs are large.
"""

input_nodes = 100 * 100
hidden_1_nodes = 256
hidden_2_nodes = 128
output_nodes = 10
learning_rate = 0.01
epochs = 200
batch_size = 64

images, labels = convert("SLD-scratch/Sign-Language-Digits-Dataset", train=True)
training_data = list(zip(images, labels))

model = SLDNNBatch(input_nodes=input_nodes, hidden_1_nodes=hidden_1_nodes, hidden_2_nodes=hidden_2_nodes, output_nodes=output_nodes, lr=learning_rate)

model.train(epochs=epochs, training_data=training_data, batch_size=batch_size)

"""
    Finally, the testing part, using the prediction function mentioned above.
"""

test_imgs, test_labels = convert("SLD-scratch/Sign-Language-Digits-Dataset", train=False)
predictions = model.test(test_imgs)

accuracy = np.mean(predictions == test_labels) * 100
print(f"NN accuracy: {accuracy:.3f}%")