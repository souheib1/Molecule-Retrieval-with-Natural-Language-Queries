"""
This script reads a log file containing training information of a deep learning model
and extracts the training and validation loss at each epoch. It then plots the learning curves.
"""


import matplotlib.pyplot as plt

# Initialize lists to store training and validation loss
train_loss = []
val_loss = []

# Open the log file
with open("log.out", "r") as file:
    lines = file.readlines()
    epoch = None
    i=0
    for line in lines:
        if line.startswith("-----EPOCH"):
            epoch = int(line.split("-----")[1].strip("EPOCH"))
            if "Validation loss:" in line:
                print(line)
                val_loss.append(float(line.split(":")[1].strip()))
        if"training loss:" in line:
            i+=1
            if i%8 == 0:
                train_loss.append(float(line.split(":")[3].strip()))
            

# Plot the learning curves
epochs= range(1, len(val_loss) + 1)
train_loss = train_loss[:-3] # adjust the size 
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
