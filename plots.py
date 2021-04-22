import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_results', type=str, default="baseline_trainVal_results.csv", help='.csv file containing baseline train and val metrics for each epoch')
parser.add_argument('--rex_results', type=str, default="rex_trainVal_results.csv", help='.csv file containing rex train and val metrics for each epoch')
parser.add_argument('--trainVal_title', type=str, default=None, help='Title of plot. Must contain model name, batch size and number of batches')
parser.add_argument('--trainVal_output', type=str, default=None, help='Plot output file name')
parser.add_argument('--compareAUC_title', type=str, default="val_auc_compare.png", help='Title for Validation AUC Comparison between Baseline and REx')
parser.add_argument('--compareAUC_output', type=str, default="val_auc_compare.png", help='Plot output file name')
parser.add_argument('--plot_name', type=str, default="compareAUC", help='Choose any of "baseline", "rex", or "compareAUC" ')

cfg = parser.parse_args()
print(cfg)

def read_file(filename):
    file = pd.read_csv(filename)
    train_losses = list(file["Train_loss"].values)
    valid_losses = list(file["Valid_loss"].values)
    valid_aucs = list(file["Valid_AUC"].values)

    return train_losses, valid_losses, valid_aucs 


class TrainValPlot():
    def __init__(self, trainVal_results, output_file, title):
        super(TrainValPlot, self).__init__()
        
        self.trainVal_results = trainVal_results
        self.output_file = output_file
        self.title = title
        
        train_losses, valid_losses, _ = read_file(self.trainVal_results)
        
        def plot_train_valid_losses(train_losses, valid_losses, output_file, title):
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
            plt.ylim(0.0, 3.0) 
            plt.xlim(0, len(train_losses)+1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            fig.savefig(filename, bbox_inches='tight')

        plot_train_valid_losses(train_losses, valid_losses, self.output_file, self.title)

class CompareValAUC():
    def __init__(self, b_results, r_results, output_file, title):
        super(CompareValAUC, self).__init__()
        
        self.baseline_results = b_results
        self.rex_results = r_results
        self.output_file = output_file
        self.title = title
        
        _, _, b_valid_aucs = read_file(self.baseline_results)
        _, _, r_valid_aucs = read_file(self.rex_results)
        
        def plot_valid_aucs(baseline, rex, output_file, title):
            # visualize the loss as the network trained
            fig = plt.figure(figsize=(10,8))
            plt.plot(range(1, len(baseline)+1), baseline, label='Baseline Valid AUCs')
            plt.plot(range(1, len(rex)+1), rex, label='REx Valid AUCs')

            plt.title(title)
            plt.xlabel('epochs')
            plt.ylabel('average tasks auc')
            plt.ylim(0.0, 1.0) 
            plt.xlim(0, len(baseline)+1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            fig.savefig(output_file, bbox_inches='tight')
            
        plot_valid_aucs(b_valid_aucs, r_valid_aucs, self.output_file, self.title)
    
    
if cfg.plot_name is "baseline":
    TrainValPlot(trainVal_results=cfg.baseline_results, output_file=cfg.trainVal_output, title=cfg.trainVal_title)
    
if cfg.plot_name is "rex":
    TrainValPlot(trainVal_results=cfg.rex_results, output_file=cfg.trainVal_output, title=cfg.trainVal_title)

if cfg.plot_name is "compareAUC":
    CompareValAUC(cfg.baseline_results, cfg.rex_results, cfg.compareAUC_output, cfg.compareAUC_title)       
    
