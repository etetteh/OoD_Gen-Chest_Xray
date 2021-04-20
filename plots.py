import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

############################################## Load and extract data ############################################
### Baseline results
baseline = pd.read_csv("baseline_trainVal_results.csv")
baseline_train_losses = list(baseline["Train_loss"].values)
baseline_valid_losses = list(baseline["Valid_loss"].values)
baseline_valid_aucs = list(baseline["Valid_AUC"].values)

### REx results
rex = pd.read_csv("rex_trainVal_results.csv")
rex_train_losses = list(rex["Train_loss"].values)
rex_valid_losses = list(rex["Valid_loss"].values)
rex_valid_aucs = list(rex["Valid_AUC"].values)


################################################ Plot Results ######################################################
def plot_train_valid_losses(train_losses, valid_losses, title=None, file_name=None):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_losses.index(min(valid_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.0, 3.0) # consistent scale
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name, bbox_inches='tight')
    
### Baseline    
file_name = 'Baseline-DenseNet-121_600B_32BS_loss_plot.png'    
title = "Baseline: Train: NIH_CHEX_MIMIM-CH Val: PC (DenseNet-121: 600B_32BS)"
plot_train_valid_losses(baseline_train_losses, baseline_valid_losses, title=title, file_name=file_name)

### REx    
file_name = 'REx-DenseNet-121_600B_32BS_loss_plot.png'    
title = "REx: Train: NIH_CHEX_MIMIM-CH Val: PC (DenseNet-121: 600B_32BS)"
plot_train_valid_losses(rex_train_losses, rex_valid_losses, title=title, file_name=file_name)


def plot_valid_aucs(baseline, rex):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, len(baseline)+1), baseline, label='Baseline Valid AUCs')
    plt.plot(range(1, len(rex)+1), rex, label='REx Valid AUCs')

    plt.title("Average AUCs - Train datasets: NIH_CHEX_MIMIM-CH | Validation dataset: PC")
    plt.xlabel('epochs')
    plt.ylabel('average tasks auc')
    plt.ylim(0.5, 1.0) # consistent scale
    plt.xlim(0, len(baseline)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('Avg_AUC_plots.png', bbox_inches='tight')
    
plot_valid_auc(baseline_valid_aucs, rex_valid_aucs)
