import warnings
from train import Train
from utils import plot_auc_curves, plot_prc_curves

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    fprs, tprs, auc, precisions, recalls, prc = Train(directory='Datasets/HMDDv3.2',
                                                      epochs=600,
                                                      in_size=64,
                                                      out_dim=64,
                                                      n_classes=64,
                                                      dropout=0.5,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=2025,
                                                      cuda=True)
    plot_auc_curves(fprs, tprs, auc, directory='plot', name='ROC_Curves')
    plot_prc_curves(precisions, recalls, prc, directory='plot', name='PR_Curves')