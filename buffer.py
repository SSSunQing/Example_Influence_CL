import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer.
    """
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'score']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, score: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param score: tensor containing the influence
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, score=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, score)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if score is not None:
                    self.score[index] = score[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, fsr=False, current_task=0) -> Tuple:
        if size > self.examples.shape[0]:
            size = self.examples.shape[0]
        if fsr and current_task > 0:
            past_examples = self.examples[self.task_labels != current_task]
            if size > past_examples.shape[0]:
                size = past_examples.shape[0]
            if past_examples.shape[0]:
                choice = np.random.choice(min(self.num_seen_examples, past_examples.shape[0]), size=size, replace=False)
                if transform is None: transform = lambda x: x
                ret_tuple = (torch.stack([transform(ee.cpu()) for ee in past_examples[choice]]).to(self.device),)
                for attr_str in self.attributes[1:]:
                    if hasattr(self, attr_str):
                        attr = getattr(self, attr_str)
                        ret_tuple += (attr[self.task_labels != current_task][choice],)
            else: return tuple([torch.tensor([0])] * 4)
        else:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=min(self.num_seen_examples, size), replace=False)
            if transform is None: transform = lambda x: x
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice],)

        return ret_tuple

    def get_data_gmed(self, size: int, transform: transforms=None, fsr=False, current_task=0) -> Tuple:
        if size > self.examples.shape[0]:
            size = self.examples.shape[0]
        if fsr and current_task > 0:
            past_examples = self.examples[self.task_labels != current_task]
            if size > past_examples.shape[0]:
                size = past_examples.shape[0]
            if past_examples.shape[0]:
                choice = np.random.choice(min(self.num_seen_examples, past_examples.shape[0]), size=size, replace=False)
                if transform is None: transform = lambda x: x
                ret_tuple = (torch.stack([transform(ee.cpu()) for ee in past_examples[choice]]).to(self.device),)
                for attr_str in self.attributes[1:]:
                    if hasattr(self, attr_str):
                        attr = getattr(self, attr_str)
                        ret_tuple += (attr[self.task_labels != current_task][choice],)
            else: return tuple([torch.tensor([0])] * 4)
        return ret_tuple + (choice,)


    def replace_data(self, index, input, label, task_id):
        if index.shape[0] != input.shape[0]:
            choice = np.random.choice(min(index.shape[0], input.shape[0]), size=min(index.shape[0], input.shape[0]), replace=False)
        if index.shape[0] > input.shape[0]:
            self.examples[index[choice]] = input.to(self.device)
            self.labels[index[choice]] = label.to(self.device)
            self.task_labels[index[choice]] = task_id.to(self.device)
        elif index.shape[0] < input.shape[0]:
            self.examples[index] = input[choice].to(self.device)
            self.labels[index] = label[choice].to(self.device)
            self.task_labels[index] = task_id[choice].to(self.device)
        else:
            self.examples[index] = input.to(self.device)
            self.labels[index] = label.to(self.device)
            self.task_labels[index] = task_id.to(self.device)


    def replace_keshihua_data(self, index, input, label, task_id, score):
        if index.shape[0] != input.shape[0]:
            choice = np.random.choice(min(index.shape[0], input.shape[0]), size=min(index.shape[0], input.shape[0]), replace=False)
        if index.shape[0] > input.shape[0]:
            self.examples[index[choice]] = input.to(self.device)
            self.labels[index[choice]] = label.to(self.device)
            self.task_labels[index[choice]] = task_id.to(self.device)
            self.score[index[choice]] = score.to(self.device)
        elif index.shape[0] < input.shape[0]:
            self.examples[index] = input[choice].to(self.device)
            self.labels[index] = label[choice].to(self.device)
            self.task_labels[index] = task_id[choice].to(self.device)
            self.score[index] = score[choice].to(self.device)
        else:
            self.examples[index] = input.to(self.device)
            self.labels[index] = label.to(self.device)
            self.task_labels[index] = task_id.to(self.device)
            self.score[index] = score.to(self.device)

    def replace_score(self, new_score, index):
        self.score[index] = new_score.to(self.device)

    def delete_data(self, num, task):
        index = []
        class_num = torch.max(self.labels).item()+1
        if num < class_num:
            num = class_num
        for i in range(class_num):
            task_ind = torch.where(self.labels == i)
            ranking = task_ind[0][:(num // class_num)].tolist()
            index = index + ranking
        index = torch.tensor(index)
        return index

    def delete_data_basedscore(self, num, task):
        index = []
        class_num = torch.max(self.labels).item() + 1
        if num < class_num:
            num = class_num
        for i in range(class_num):
            task_ind = torch.where(self.labels == i)
            a = torch.sort(self.score[task_ind[0]][:, 2], descending=True)
            ranking = task_ind[0][a[1]][:(num // class_num)].tolist()
            index = index + ranking
        index = torch.tensor(index)
        return index


    def reset_score(self):
        new_score = torch.ones((self.examples.shape[0], 3), device=self.device)
        self.score = new_score