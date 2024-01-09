import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
import torch.nn.init as init 
from matplotlib import pyplot as plt
import torch.nn.functional as F
import wandb 
from scipy.stats import binom_test
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from sklearn.model_selection import train_test_split
from collections import Counter
from mamba_simple import Block, Mamba

# Define the device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Making a class that uses the Mamba architecture for a classification task
class MambaClass(nn.Module):
    def __init__(self, num_blocks, d_model, num_classes):
        super(MambaClass, self).__init__()

        # Modify the lambda to accept an argument and pass it to Mamba
        mamba_creator = lambda dim: Mamba(d_model=dim)

        self.layers = nn.ModuleList([
            Block(dim=d_model, mixer_cls=mamba_creator)
            for _ in range(num_blocks)
        ])

        # Use a dummy input to determine the output size after the blocks
        with torch.no_grad():
             # Ensure the dummy input is on the same device as the model
            dummy_input = torch.zeros(1, 145, d_model).float().to(device)  # [batch, time series, channels]
            dummy_output = self.forward_blocks(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        # Now create the classifier with the correct input size
        self.classifier = nn.Linear(flattened_size, num_classes)

    def forward_blocks(self, x):
        # Function to forward the input through the blocks
        for block in self.layers:
            output, _ = block(x)
            x = output
        return x

    def forward(self, x):
        x = self.forward_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        final_output = self.classifier(x)
        if not self.training:
            final_output = F.softmax(final_output, dim=1)
        return final_output


# Load data and labels
train_data = np.load(r'/mnt/c/Users/eddyt/Desktop/imagined speech classification/mamba/data/388 channels/train-388.npy')
train_labels = np.load(r'/mnt/c/Users/eddyt/Desktop/imagined speech classification/mamba/data/388 channels/train-label-388.npy')

test_data = np.load(r'/mnt/c/Users/eddyt/Desktop/imagined speech classification/mamba/data/388 channels/test-388.npy')
test_labels = np.load(r'/mnt/c/Users/eddyt/Desktop/imagined speech classification/mamba/data/388 channels/test-label-388.npy')

# Add a parameter for specifying the directory to save the weights
#save_dir = r'C:\Users\eddyt\Desktop\Semantic_decoder_Git\semantic-decoding\processed_data\weights'  # Set your desired path here
#os.makedirs(save_dir, exist_ok=True)

# Function to save weights of the first few layers
def save_layer_weights(model, epoch, iteration, save_dir):
    layer_weights = {}
    for name, param in model.named_parameters():
        if 'conv1' in name or 'conv2' in name:  # Focusing on conv1 and conv2
            layer_weights[name] = param.detach().cpu().numpy()

    save_path = os.path.join(save_dir, f"weights_epoch_{epoch}_iter_{iteration}.npz")
    np.savez(save_path, **layer_weights)
    print(f"Weights saved at {save_path}")


print("Training Data Info #########")
print("Data Shape:", train_data.shape)
print("Data Type:", train_data.dtype)
print("Labels:", train_labels)
print("Labels Type:", train_labels.dtype)
# Count the number of each label in the labels array
unique, counts = np.unique(train_labels, return_counts=True)
label_counts = dict(zip(unique, counts))
for label, count in label_counts.items():
    print(f"Label {label} has {count} datasets")

print("Testing Data Info #########")
print("Data Shape:", test_data.shape)
print("Data Type:", test_data.dtype)
print("Labels:", test_labels)
print("Labels Type:", test_labels.dtype)
unique, counts = np.unique(test_labels, return_counts=True)
label_counts = dict(zip(unique, counts))
for label, count in label_counts.items():
    print(f"Label {label} has {count} datasets")

# Split the data into training and testing sets
# train_data, test_data, train_labels, test_labels = train_test_split(
#     data, labels, test_size=0.2, random_state=None, stratify=labels
# )
print("Unique Labels in Training Set:", np.unique(train_labels))
print("Unique Labels in Validation Set:", np.unique(test_labels))

# Define a custom Dataset
class NIRS_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create datasets and dataloaders for training and testing
train_dataset = NIRS_Dataset(train_data, train_labels)
test_dataset = NIRS_Dataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)


num_blocks = 5 # number of mamba blocks 
d_model = 388  # dimensionality matches the number of channels
num_classes= 3 # number of classes in the classification task

model = MambaClass(num_blocks=num_blocks, d_model=d_model, num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define the number of splits for cross-validation
num_epochs=50   #for default testing use 50
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#setup project name for W&B
#wandb.init(project="Mamba")
#wandb.config.num_epochs = num_epochs
#wandb.config.batch_size = 8
#wandb.config.learning_rate = 0.001


# Initialize accumulators
total_accuracy= 0
num_iterations = 50  #for normal testing use 50

# Initialize accumulators
iteration_accuracies = []  # List to store accuracy of each iteration
rolling_averages = []  # List to store rolling averages

# Repeat training and testing num_iterations times
for iteration in range(num_iterations):
    print(f"Iteration: {iteration + 1}/{num_iterations}")

    # Reinitialize model and optimizer
    model = MambaClass(num_blocks=num_blocks, d_model=d_model, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.long().to(device) - 1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate and print average loss per batch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}")

        # Save weights every 5 epochs
        #if epoch % 5 == 0:
        #    save_layer_weights(model, epoch, iteration, save_dir)

    # Testing loop
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device) - 1

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Print model predictions and actual labels for comparison
            print(f"Model predictions: {predicted}")
            print(f"Actual labels: {labels}")

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate accuracy for this iteration
    accuracy = correct_predictions / total_predictions
    print(f'Iteration {iteration + 1} Accuracy: {accuracy}')
    iteration_accuracies.append(accuracy)
    total_accuracy= total_accuracy+accuracy

    # Calculate rolling average
    rolling_average = sum(iteration_accuracies) / (iteration + 1)
    rolling_averages.append(rolling_average)
    print(f'Rolling Average Accuracy after {iteration + 1} iterations: {rolling_average}')

# Calculate and print average accuracy over 50 iterations
average_accuracy = total_accuracy / num_iterations
print(f'Average Accuracy over {num_iterations} iterations: {average_accuracy}')

# Plot the rolling average accuracy against iteration
plt.plot(range(1, num_iterations + 1), rolling_averages)
plt.xlabel('Iteration')
plt.ylabel('Rolling Average Accuracy')
plt.title('Rolling Average Accuracy per Iteration')
plt.show()

# Finalize wandb run
#wandb.finish()


