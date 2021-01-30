import time

import torch
import torch.optim as optim
from utils import get_memory_and_execution_time_details

from train_test_split import train_test_split
from KnowledgeDistillation import KD
from src.NeuralNetworks.GCN import GCN
from src.NeuralNetworks.RecurrentGCN import RecurrentGCN

train_loader, test_loader, num_node_features = train_test_split()

student_model = GCN(num_node_features=num_node_features, hidden_channels=[100])

lr = 10e-5
weight_decay = 5e-4

student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

# Distilled EvolveGCN

epochs = 500

teacher_model_evolvegcn = RecurrentGCN(node_features=num_node_features, num_classes=2)
teacher_model_evolvegcn.load_state_dict(torch.load("./models/evolvegcn_teacher.pt"))

teacher_optimizer_evolvegcn = optim.Adam(teacher_model_evolvegcn.parameters(), lr=lr, weight_decay=weight_decay,
                                         amsgrad=True)

student_model.load_state_dict(torch.load("./models/evolvegcn_student.pt"))

distiller_evolvegcn = KD.VanillaKD(teacher_model_evolvegcn, student_model, train_loader, test_loader,
                                   teacher_optimizer_evolvegcn, student_optimizer)

get_memory_and_execution_time_details(distiller_evolvegcn.evaluate, True)  # Evaluate the teacher network

get_memory_and_execution_time_details(distiller_evolvegcn.evaluate, False)  # Evaluate the student network

distiller_evolvegcn.get_parameters()

# Distilled Deep Neural Decision Forest

from src.DeepNeuralDecisionForest import NeuralDecisionForest as ndf

epochs = 100

feat_layer = RecurrentGCN(node_features=num_node_features, num_classes=2, dropout_rate=0.65)
forest = ndf.Forest(n_tree=80, tree_depth=8, n_class=2, n_in_feature=2, tree_feature_rate=0.65)
teacher_model_dndf = ndf.NeuralDecisionForest(feat_layer, forest)

teacher_model_dndf.load_state_dict(torch.load("./models/dndf_teacher.pt"))

teacher_optimizer_dndf = optim.Adam(teacher_model_dndf.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
student_model.load_state_dict(torch.load("./models/dndf_student.pt"))

distiller_dndf = KD.VanillaKD(teacher_model_dndf, student_model, train_loader, test_loader,
                              teacher_optimizer_dndf, student_optimizer)

get_memory_and_execution_time_details(distiller_dndf.evaluate, True)  # Evaluate the teacher network

get_memory_and_execution_time_details(distiller_dndf.evaluate, False)  # Evaluate the student network

distiller_dndf.get_parameters()
