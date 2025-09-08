import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import interp
from numpy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
plt.rcParams["font.family"] = "Times New Roman"


#绘制ROC曲线
def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    #plt.savefig(directory + '/%s.jpg' % name, dpi=600, bbox_inches='tight', format="svg")
    plt.savefig(directory + '/%s.jpg' % name, dpi=600, bbox_inches='tight')
    plt.close()
#PR曲线代码
def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9, label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))
    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    #plt.savefig(directory + '/%s.jpg' % name, dpi=600, bbox_inches='tight',format="svg")
    plt.savefig(directory + '/%s.jpg' % name, dpi=600, bbox_inches='tight')
    plt.close()

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()
#CKA-MKL
def load_data(directory):
    # 加载疾病和miRNA的相似性矩阵
    D_SSM1 = np.loadtxt(directory + '/diseaseSim/DiseaseSimilarity1.txt')  # 疾病语义相似性1
    D_SSM2 = np.loadtxt(directory + '/diseaseSim/DiseaseSimilarity2.txt')  # 疾病语义相似性2
    D_GSM = np.loadtxt(directory + '/diseaseSim/D_GIP.txt')  # 疾病GIP核相似性

    np.fill_diagonal(D_SSM1, 1)
    np.fill_diagonal(D_SSM2, 1)

    D_SSM = (D_SSM1 + D_SSM2)/2

    M_FSM = np.loadtxt(directory + '/miRNASim/FuncSim.txt')  # miRNA功能相似性
    M_SeSM = np.loadtxt(directory + '/miRNASim/SeqSim.txt')  # miRNA序列相似性
    M_Fam = np.loadtxt(directory + '/miRNASim/Famsim.txt')  # miRNA家族相似性
    M_GSM = np.loadtxt(directory + '/miRNASim/M_GIP.txt')  # miRNA GIP核相似性

    # miRNA功能相似性融合家族信息
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_Fam[i][j] == 1:
                M_FSM[i][j] = (M_FSM[i][j] + M_Fam[i][j]) / 2
            else:
                M_FSM[i][j] = M_FSM[i][j]

    all_associations = pd.read_csv(directory + '/new_adjacency_matrix.csv', names=['miRNA', 'disease', 'label'])
    return (D_SSM, D_GSM), (M_FSM, M_SeSM, M_GSM), all_associations


def build_graph(directory, random_seed):
    ID, IM, samples = preprocess(directory, random_seed)
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    disease_ids = list(range(1, ID.shape[0] + 1))
    mirna_ids = list(range(1, IM.shape[0] + 1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    #g.readonly()
    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples






