import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import numpy as np

def plot_epochs(train_data, test_data, epochs_lim=100, metric='Average Loss'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for data, label in [(train_data, 'Train'), (test_data, 'Test')]:
        epochs = data['Epoch'][:epochs_lim]       
        values = data[metric][:epochs_lim]
        ax.plot(epochs, values, label=f'{label}: {metric}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'{metric} over Epochs')
    ax.legend()
    

def visualize_metrics(train_data, test_data, path=None, epochs_lim=100, metric= 'Average Loss', title= 'Average Loss'):
    sns.set(style='darkgrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for data, label, color in [(train_data, 'Train', 'blue'), (test_data, 'Test', 'red')]:
        melted_data = data.melt(id_vars='Epoch', value_vars=metric, var_name='Metric', value_name='Value')
        
        sns.lineplot(data=melted_data[:epochs_lim], x='Epoch', y='Value', ax=ax, label=label, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{title}')
    ax.set_title(f'{title} over Epochs')
    ax.legend()
    
    if path:
        fig.savefig(path)

def compare_models(scores_per_model, path=None, epochs_lim=100, metric= 'Average Loss', title= 'Average Loss'):
    sns.set(style='darkgrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, label, color in scores_per_model:
        # Find the index of the maximum value in y
        max_x = np.argmax(data[metric])
        max_y = np.max(data[metric])

        melted_data = data.melt(id_vars='Epoch', value_vars=metric, var_name='Metric', value_name='Value')
        
        sns.lineplot(data=melted_data[:epochs_lim], x='Epoch', y='Value', ax=ax, label=label, color=color)
        # Add a marker at the maximum point
        # plt.scatter(max_x, max_y, color='black', s=10)
        # plt.text(max_x, max_y, str(round(max_y, 2)), color=color,  ha='center', va='bottom')


    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{title}')
    ax.set_title(f'{title} over Epochs')
    ax.legend()
    
    if path:
        fig.savefig(path)




if __name__ == '__main__':
    # Read train and test data from CSV files
    # train_data = pd.read_csv('models/results/100X/train/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25.csv')
    # test_data = pd.read_csv('models/results/100X/test/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25.csv')

    # m_titles = {
    # 'accuracy_score': 'Accuracy',
    # 'roc_auc_score': 'ROC AUC',
    # 'average_precision_score' : 'Average Precision',
    # 'f1_score' : 'F1 Score',
    # 'recall_Score' : 'Recall',
    # 'cohen_kappa_score' : 'Cohen-Kappa Score',
    # 'specificity_score': 'Specificity'
    # }

    # for metric, title in m_titles.items():
    #     visualize_metrics(train_data=train_data, test_data=test_data, 
    #                             path=f'models/results/100X/figs/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25_{metric}.png',
    #                             metric=metric,
    #                             title=title)

    # plt.show()


    path = 'models/results/40X/test'
    files = os.listdir(path)
        # 1- models\results\40X\weights\40X_on-air-aug_std_none_pre-resnet18_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-23.pth
    # 2- models\arch_results\40X\weights\40X_none_std_none_pre-resnet18_sgde-2_bce_32bs-strf_100ep_2023-06-19.pth
    # 3- models\arch_results\40X\weights\40X_on-air-sp_std_none_pre-resnet18_sgde-2_bce_32bs-strf_100ep_2023-06-16.pth

    # files = ["models/arch_results/40X/test/40X_none_std_none_pre-resnet18_sgde-2_bce_32bs-strf_100ep_2023-06-19.csv",
    # "models/arch_results/40X/test/40X_on-air-sp_std_none_pre-resnet18_sgde-2_bce_32bs-strf_100ep_2023-06-16.csv",
    # "models/results/40X/test/40X_on-air-aug_std_none_pre-resnet18_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-23.csv"]
    scores_per_model = []

    # colors = ['green', 'red', 'blue']
    # labels = ['No Augmentation-Patching', '4-Tile Patching', 'Augmentation']

    colors = ['green', 'red', 'blue', 'purple', 'orange']
    labels = ['FPN', 'GoogleNet', 'MnasNet', 'MobileNet', 'Resnet18']

    for i, file in enumerate(files):
        print(file)
        data = pd.read_csv(os.path.join(path,file))
        scores_per_model.append((data, labels[i], colors[i]))

    compare_models(scores_per_model, metric='accuracy_score', title='Accuracy',
                   path= 'models/results/40X/figs/40X_max_acc.png')

