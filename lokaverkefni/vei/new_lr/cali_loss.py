import json
import matplotlib.pyplot as plt

# Load the JSON file containing the results
with open('cali_mega_train.json', 'r') as json_file:
    results = json.load(json_file)

# Extract training losses from the loaded data
train_losses = results['train_losses']

# Plot the training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
