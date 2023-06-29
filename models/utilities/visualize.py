import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        melted_data = data.melt(id_vars='Epoch', value_vars=metric, var_name='Metric', value_name='Value')
        
        sns.lineplot(data=melted_data[:epochs_lim], x='Epoch', y='Value', ax=ax, label=label, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{title}')
    ax.set_title(f'{title} over Epochs')
    ax.legend()
    
    if path:
        fig.savefig(path)




if __name__ == '__main__':
    # Read train and test data from CSV files
    train_data = pd.read_csv('models/results/100X/train/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25.csv')
    test_data = pd.read_csv('models/results/100X/test/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25.csv')

    m_titles = {
    'accuracy_score': 'Accuracy',
    'roc_auc_score': 'ROC AUC',
    'average_precision_score' : 'Average Precision',
    'f1_score' : 'F1 Score',
    'recall_Score' : 'Recall',
    'cohen_kappa_score' : 'Cohen-Kappa Score',
    'specificity_score': 'Specificity'
    }

    for metric, title in m_titles.items():
        visualize_metrics(train_data=train_data, test_data=test_data, 
                                path=f'models/results/100X/figs/100X_on-air-aug_std_weightrandom_pre-fpcnv2_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-25_{metric}.png',
                                metric=metric,
                                title=title)

    # plt.show()


    # path = 'models/results/200X/test'
    # files = os.listdir(path)
    # scores_per_model = []

    # colors = ['blue', 'red', 'green', 'pink', 'orange']
    # labels = ['FPN', 'GoogleNet', 'MnasNet', 'MobileNet', 'Resnet18']

    # for i, file in enumerate(files):
    #     print(file)
    #     data = pd.read_csv(os.path.join(path, file))
    #     scores_per_model.append((data, labels[i], colors[i]))

    # compare_models(scores_per_model, metric='specificity_score', title='Specificity',
    #                path= 'models/results/200X/figs/200X_test_comparison_specificity_score.png')