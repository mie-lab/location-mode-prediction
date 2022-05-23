import torch
import numpy as np


class baselines:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.persistent_forcast()
        # self.frequent_forcast()
        self.markov_forcast_local()

    def persistent_forcast(self):
        """Predict the same value as the previous timestep (data==target, lower bound)."""
        correct = 0
        total = 0

        for _, (_, _, target, dict) in enumerate(self.train_loader):
            data = dict["mode"]
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forcast train accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(self.val_loader):
            data = dict["mode"]
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forcast validation accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(self.test_loader):
            data = dict["mode"]
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forcast test accuracy = {:.2f}".format(100 * correct / total))

    def frequent_forcast(self):
        """Predict the most frequent value as the target."""
        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(self.train_loader):
            data = dict["mode"]
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forcast train accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(self.val_loader):
            data = dict["mode"]
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forcast validation Accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(self.test_loader):
            data = dict["mode"]
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forcast test Accuracy = {:.2f}".format(100 * correct / total))

    def markov_forcast_self(self, loader):
        correct = 0
        total = 0
        for _, (_, _, target, dict) in enumerate(loader):
            data = dict["mode"]
            # most frequent
            # output, counts = torch.unique(data, sorted=False, return_counts=True)
            # most_frequent = output[counts.argmax()]

            # persistent
            most_frequent = data[-1]

            # markov matrix construction
            idx, inverse_indices = data.unique(return_inverse=True)
            mark_matrix = np.zeros([idx.shape[0], idx.shape[0]])
            for i in range(data.shape[0] - 1):
                mark_matrix[inverse_indices[i], inverse_indices[i + 1]] += 1

            current_predict = np.argmax(mark_matrix[inverse_indices[-1], :])
            if current_predict != 0:
                current_predict = idx[current_predict]
            else:
                current_predict = most_frequent

            correct += (current_predict == target).numpy()[0]
            total += 1

        print("Accuracy = {:.2f}".format(100 * correct / total))

    def markov_forcast_local(self):
        """Predict according to the previous experience, otherwise predict the most frequent value as the target."""
        self.markov_forcast_self(self.train_loader)
        self.markov_forcast_self(self.val_loader)
        self.markov_forcast_self(self.test_loader)
