import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected

from KnowledgeDistillation import KD
from src.NeuralNetworks.GCN import GCN
from src.NeuralNetworks.RecurrentGCN import RecurrentGCN

# Load Dataframe
df_edge = pd.read_csv('../datasets/elliptic_txs_edgelist.csv')
df_class = pd.read_csv('../datasets/elliptic_txs_classes.csv')
df_features = pd.read_csv('../datasets/elliptic_txs_features.csv', header=None)

# Setting Column name
df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                      range(72)]

print('Number of edges: {}'.format(len(df_edge)))
df_edge.head()

# Get Node Index

all_nodes = list(
    set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
nodes_df = pd.DataFrame(all_nodes, columns=['id']).reset_index()

print('Number of nodes: {}'.format(len(nodes_df)))
nodes_df.head()

# Fix id index

df_edge = df_edge.join(nodes_df.rename(columns={'id': 'txId1'}).set_index('txId1'), on='txId1', how='inner') \
    .join(nodes_df.rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='inner', rsuffix='2') \
    .drop(columns=['txId1', 'txId2']) \
    .rename(columns={'index': 'txId1', 'index2': 'txId2'})
df_edge.head()

df_class = df_class.join(nodes_df.rename(columns={'id': 'txId'}).set_index('txId'), on='txId', how='inner') \
    .drop(columns=['txId']).rename(columns={'index': 'txId'})[['txId', 'class']]
df_class.head()

df_features = df_features.join(nodes_df.set_index('id'), on='id', how='inner') \
    .drop(columns=['id']).rename(columns={'index': 'id'})
df_features = df_features[['id'] + list(df_features.drop(columns=['id']).columns)]
df_features.head()

df_edge_time = df_edge.join(df_features[['id', 'time step']].rename(columns={'id': 'txId1'}).set_index('txId1'),
                            on='txId1', how='left', rsuffix='1') \
    .join(df_features[['id', 'time step']].rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='left',
          rsuffix='2')
df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
df_edge_time_fin = df_edge_time[['txId1', 'txId2', 'time step']].rename(
    columns={'txId1': 'source', 'txId2': 'target', 'time step': 'time'})

# Create csv from Dataframe

df_features.drop(columns=['time step']).to_csv('../datasets_cont/elliptic_txs_features.csv', index=False, header=None)
df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').to_csv(
    '../datasets_cont/elliptic_txs_classes.csv', index=False, header=None)
df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'})[['nid', 'time']].sort_values(
    by='nid').to_csv('../datasets_cont/elliptic_txs_nodetime.csv', index=False, header=None)
df_edge_time_fin[['source', 'target', 'time']].to_csv('../datasets_cont/elliptic_txs_edgelist_timed.csv', index=False,
                                                      header=None)

# Graph Preprocessing

node_label = df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').merge(
    df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'}), on='nid', how='left')
node_label['label'] = node_label['label'].apply(lambda x: '3' if x == 'unknown' else x).astype(int) - 1
node_label.head()

merged_nodes_df = node_label.merge(
    df_features.rename(columns={'id': 'nid', 'time step': 'time'}).drop(columns=['time']), on='nid', how='left')
merged_nodes_df.head()

train_dataset = []
test_dataset = []

num_node_features = 0
for i in range(49):
    nodes_df_tmp = merged_nodes_df[merged_nodes_df['time'] == i + 1].reset_index()
    nodes_df_tmp['index'] = nodes_df_tmp.index
    df_edge_tmp = df_edge_time_fin.join(
        nodes_df_tmp.rename(columns={'nid': 'source'})[['source', 'index']].set_index('source'), on='source',
        how='inner') \
        .join(nodes_df_tmp.rename(columns={'nid': 'target'})[['target', 'index']].set_index('target'), on='target',
              how='inner', rsuffix='2') \
        .drop(columns=['source', 'target']) \
        .rename(columns={'index': 'source', 'index2': 'target'})
    x = torch.tensor(np.array(nodes_df_tmp.sort_values(by='index').drop(columns=['index', 'nid', 'label'])),
                     dtype=torch.float)
    edge_index = torch.tensor(np.array(df_edge_tmp[['source', 'target']]).T, dtype=torch.long)
    edge_index = to_undirected(edge_index)
    mask = nodes_df_tmp['label'] != 2
    y = torch.tensor(np.array(nodes_df_tmp['label']), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, mask=mask, y=y)
    num_node_features = data.num_node_features
    if i + 1 < 35:
        train_dataset.append(data)
    else:
        test_dataset.append(data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

student_model = GCN(num_node_features=num_node_features, hidden_channels=[100])

lr = 10e-5
weight_decay = 5e-4

student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

# Distilled EvolveGCN
#
# epochs = 500
#
# teacher_model_evolvegcn = RecurrentGCN(node_features=num_node_features, num_classes=2)
#
# # teacher_model_evolvegcn.load_state_dict(torch.load("./models/evolvegcn_teacher.pt"))
#
# teacher_optimizer_evolvegcn = optim.Adam(teacher_model_evolvegcn.parameters(), lr=lr, weight_decay=weight_decay,
#                                          amsgrad=True)
#
# distiller_evolvegcn = KD.VanillaKD(teacher_model_evolvegcn, student_model, train_loader, test_loader,
#                                    teacher_optimizer_evolvegcn, student_optimizer)
# distiller_evolvegcn.train_teacher(epochs=epochs, plot_losses=True, save_model=True,
#                                   save_model_pth='./models/evolvegcn_teacher.pt'
#                                   # ,save_plot_pth="./outputs/evolvegcn_teacher.png"
#                                   )  # Train the teacher network
# start_time = time.time()
# distiller_evolvegcn.evaluate(teacher=True)  # Evaluate the teacher network
# evolvegcn_teacher_time = time.time() - start_time
# print("Teacher Model Evaluation Time: ")
# print(evolvegcn_teacher_time)
#
# distiller_evolvegcn.train_student(epochs=epochs, plot_losses=True, save_model=True,
#                                   save_model_pth='./models/evolvegcn_student.pt'
#                                   # , save_plot_pth="./outputs/evolvegcn_student.png"
#                                   )  # Train the student network
# start_time = time.time()
# distiller_evolvegcn.evaluate(teacher=False)  # Evaluate the teacher network
# evolvegcn_student_time = time.time() - start_time
# print("Student Model Evaluation Time: ")
# print(evolvegcn_student_time)
#
# distiller_evolvegcn.get_parameters()

# Distilled Deep Neural Decision Forest

from src.DeepNeuralDecisionForest import NeuralDecisionForest as ndf

epochs = 100

feat_layer = RecurrentGCN(node_features=num_node_features, num_classes=2, dropout_rate=0.65)
forest = ndf.Forest(n_tree=80, tree_depth=8, n_class=2, n_in_feature=2, tree_feature_rate=0.65)
teacher_model_dndf = ndf.NeuralDecisionForest(feat_layer, forest)

teacher_optimizer_dndf = optim.Adam(teacher_model_dndf.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

distiller_dndf = KD.VanillaKD(teacher_model_dndf, student_model, train_loader, test_loader,
                              teacher_optimizer_dndf, student_optimizer)
distiller_dndf.train_teacher(epochs=epochs, plot_losses=True, save_model=True,
                             save_model_pth='./models/dndf_teacher.pt'
                             # , save_plot_pth="./outputs/dndf_teacher.png"
                             )  # Train the teacher network
start_time = time.time()
distiller_dndf.evaluate(teacher=True)  # Evaluate the teacher network
dndf_teacher_time = time.time() - start_time
print("Teacher Model Evaluation Time: ")
print(dndf_teacher_time)

distiller_dndf.train_student(epochs=epochs, plot_losses=True, save_model=True,
                             save_model_pth='./models/dndf_student.pt'
                             # , save_plot_pth="./outputs/dndf_student.png"
                             )  # Train the student network
start_time = time.time()
distiller_dndf.evaluate(teacher=False)  # Evaluate the teacher network
dndf_student_time = time.time() - start_time
print("Student Model Evaluation Time: ")
print(dndf_student_time)

distiller_dndf.get_parameters()
