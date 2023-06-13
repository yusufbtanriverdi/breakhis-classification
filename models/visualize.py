import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_epochs(train_data, test_data, metric='Average Loss'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for data, label in [(train_data, 'Train'), (test_data, 'Test')]:
        epochs = data['Epoch']        
        values = data[metric]
        ax.plot(epochs, values, label=f'{label}: {metric}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'{metric} over Epochs')
    ax.legend()
    
    plt.show()


def visualize_metrics(train_data, test_data, metric= 'Average Loss'):
    sns.set(style='darkgrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for data, label, color in [(train_data, 'Train', 'blue'), (test_data, 'Test', 'red')]:
        melted_data = data.melt(id_vars='Epoch', value_vars=metric, var_name='Metric', value_name='Value')
        
        sns.lineplot(data=melted_data, x='Epoch', y='Value', ax=ax, label=label, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'{metric} over Epochs')
    ax.legend()
    
    plt.show()


if __name__ == '__main__':
    # Read train and test data from CSV files
    train_data = pd.read_csv('models/results/train/ResNet_2023-06-13.csv')
    test_data = pd.read_csv('models/results/test/ResNet_2023-06-13.csv')

    plot_epochs(train_data, test_data, metric='accuracy_score')
    visualize_metrics(train_data, test_data, metric='accuracy_score')
