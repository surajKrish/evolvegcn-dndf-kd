import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter


class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
            self,
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn=nn.KLDivLoss(),
            temp=20.0,
            distil_weight=0.5,
            device="cpu",
            log=False,
            logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        try:
            torch.Tensor(0).to(device)
            self.device = device
        except:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")

        try:
            self.teacher_model = teacher_model.to(self.device)
        except:
            print("Warning!!! Teacher is NONE.")
        self.student_model = student_model.to(self.device)
        try:
            self.loss_fn = loss_fn.to(self.device)
            self.ce_fn = nn.CrossEntropyLoss().to(self.device)
        except:
            self.loss_fn = loss_fn
            self.ce_fn = nn.CrossEntropyLoss()
            print("Warning: Loss Function can't be moved to device.")

    def train_teacher(
            self,
            epochs=20,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/teacher.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.teacher_model.train()
        loss_arr = []
        illicit_f1_arr = []
        micro_avg_f1_arr = []
        illicit_precision_arr = []
        micro_avg_precision_arr = []
        illicit_recall_arr = []
        micro_avg_recall_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Teacher... ")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            torch.manual_seed(ep)
            np.random.seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for data in self.train_loader:

                data.x = data.x.to(self.device)
                label = data.y.to(self.device)
                mask = data.mask

                out = self.teacher_model(data)

                if isinstance(out, tuple):
                    out = out[0]

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                illicit_f1_arr.append(f1_score(pred[mask], label[mask], pos_label=1))
                micro_avg_f1_arr.append(f1_score(pred[mask], label[mask], average='micro'))
                illicit_precision_arr.append(precision_score(pred[mask], label[mask], pos_label=1))
                micro_avg_precision_arr.append(precision_score(pred[mask], label[mask], average='micro'))
                illicit_recall_arr.append(recall_score(pred[mask], label[mask], pos_label=1))
                micro_avg_recall_arr.append(recall_score(pred[mask], label[mask], average='micro'))

                loss = self.ce_fn(out[mask], label[mask])

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(
                'Epoch: {:1d}, Epoch Loss: {:.4f}, Epoch Accuracy: {:.4f}, Illicit Precision: {:.4f}, Illicit Recall: '
                '{:.4f}, Illicit f1: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}' \
                    .format(ep + 1, epoch_loss, epoch_acc, np.mean(illicit_precision_arr),
                            np.mean(illicit_recall_arr), np.mean(illicit_f1_arr), np.mean(micro_avg_f1_arr),
                            np.mean(micro_avg_precision_arr), np.mean(micro_avg_recall_arr)))

            self.post_epoch_call(ep)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def _train_student(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/student.pt",
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        illicit_f1_arr = []
        micro_avg_f1_arr = []
        illicit_precision_arr = []
        micro_avg_precision_arr = []
        illicit_recall_arr = []
        micro_avg_recall_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            torch.manual_seed(ep)
            np.random.seed(ep)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for data in self.train_loader:

                data.x = data.x.to(self.device)
                label = data.y.to(self.device)
                mask = data.mask

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)

                loss = self.calculate_kd_loss(student_out[mask], teacher_out[mask], label[mask])

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                illicit_f1_arr.append(f1_score(pred[mask], label[mask], pos_label=1))
                micro_avg_f1_arr.append(f1_score(pred[mask], label[mask], average='micro'))
                illicit_precision_arr.append(precision_score(pred[mask], label[mask], pos_label=1))
                micro_avg_precision_arr.append(precision_score(pred[mask], label[mask], average='micro'))
                illicit_recall_arr.append(recall_score(pred[mask], label[mask], pos_label=1))
                micro_avg_recall_arr.append(recall_score(pred[mask], label[mask], average='micro'))

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(
                'Epoch: {:1d}, Epoch Loss: {:.4f}, Epoch Accuracy: {:.4f}, Illicit Precision: {:.4f}, Illicit Recall: '
                '{:.4f}, Illicit f1: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}' \
                    .format(ep + 1, epoch_loss, epoch_acc, np.mean(illicit_precision_arr),
                            np.mean(illicit_recall_arr), np.mean(illicit_f1_arr), np.mean(micro_avg_f1_arr),
                            np.mean(micro_avg_precision_arr), np.mean(micro_avg_recall_arr)))

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/student.pt",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self._train_student(epochs, plot_losses, save_model, save_model_pth)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=False):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []
        illicit_f1_arr = []
        micro_avg_f1_arr = []
        illicit_precision_arr = []
        micro_avg_precision_arr = []
        illicit_recall_arr = []
        micro_avg_recall_arr = []

        seed_val = 35

        with torch.no_grad():
            for data in self.train_loader:

                torch.manual_seed(seed_val)
                np.random.seed(seed_val)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                data.x = data.x.to(self.device)
                target = data.y.to(self.device)
                mask = data.mask

                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / length_of_dataset
                illicit_f1_arr.append(f1_score(pred[mask], target[mask], pos_label=1))
                micro_avg_f1_arr.append(f1_score(pred[mask], target[mask], average='micro'))
                illicit_precision_arr.append(precision_score(pred[mask], target[mask], pos_label=1))
                micro_avg_precision_arr.append(precision_score(pred[mask], target[mask], average='micro'))
                illicit_recall_arr.append(recall_score(pred[mask], target[mask], pos_label=1))
                micro_avg_recall_arr.append(recall_score(pred[mask], target[mask], average='micro'))

                if verbose:
                    print("-" * 80)
                    print(f"Iteration: {seed_val-34}")
                    print("-" * 80)
                    print("Illicit F1: {:.4f}".format(f1_score(pred[mask], target[mask], pos_label=1)))
                    print("Illicit Precision: {:.4f}".format(precision_score(pred[mask], target[mask], pos_label=1)))
                    print("Illicit Recall: {:.4f}".format(recall_score(pred[mask], target[mask], pos_label=1)))
                    print("Micro Avg F1: {:.4f}".format(f1_score(pred[mask], target[mask], average='micro')))
                    print("Micro Avg Precision: {:.4f}".format(precision_score(pred[mask], target[mask], average='micro')))
                    print("Micro Avg Recall: {:.4f}".format(recall_score(pred[mask], target[mask], average='micro')))

                seed_val += 1

        print("-" * 80)
        print("-" * 80)
        print("Final Result")
        print("-" * 80)
        print("-" * 80)
        print(f"Accuracy: {accuracy}")
        print("Illicit F1: {:.4f}".format(np.mean(illicit_f1_arr)))
        print("Illicit Precision: {:.4f}".format(np.mean(illicit_precision_arr)))
        print("Illicit Recall: {:.4f}".format(np.mean(illicit_recall_arr)))
        print("Micro Avg F1: {:.4f}".format(np.mean(micro_avg_f1_arr)))
        print("Micro Avg Precision: {:.4f}".format(np.mean(micro_avg_precision_arr)))
        print("Micro Avg Recall: {:.4f}".format(np.mean(micro_avg_recall_arr)))
        return outputs, accuracy

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy = self._evaluate_model(model=model, verbose=False)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print(f"Total parameters for the teacher network are: {teacher_params}")
        print(f"Total parameters for the student network are: {student_params}")

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
