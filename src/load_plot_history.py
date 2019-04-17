import numpy as np 
import matplotlib.pyplot as plt

def get_history(file_path):
    model_history = np.load(file_path).item()
    metrics = ['loss', 'acc', 'val_loss', 'val_acc']

    for metric in metrics:
        y_vals = model_history.history[metric]
        plt.plot(y_vals, label=metric)

    plt.legend()
    plt.show()

def compare_models(file_path_model1, file_path_model2, file_path_model3, metric='loss'):
    model_1_history = np.load(file_path_model1).item()
    model_2_history = np.load(file_path_model2).item()
    model_3_history = np.load(file_path_model3).item()

    metrics = ['loss', 'acc', 'val_loss', 'val_acc']

    model_1 = model_1_history.history[metric]
    model_2 = model_2_history.history[metric]
    model_3 = model_3_history.history[metric]
    
    model_loss_vals = [model_1, model_2, model_3]

    model_names = ['1000_20', '1000_30', '1000_40']

    for idx, model_loss_val in enumerate(model_loss_vals):
        y_vals = model_loss_val
        plt.plot(y_vals, label=model_names[idx])

    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path_model1 = 'models/jerry/samples_1000_seq_20/jerry_model_history.npy'
    file_path_model2 = 'models/jerry/samples_1000_seq_30/jerry_model_history.npy'
    file_path_model3 = 'models/jerry/samples_1000_seq_40/jerry_model_history.npy'
    #get_history(file_path_model3)
    compare_models(file_path_model1, file_path_model2, file_path_model3, metric='val_acc')