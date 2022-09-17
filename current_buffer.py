import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from utils.ourskmeans import cluster


def reservoir(num_examples: int, buffer_size: int) -> int:
    if num_examples < buffer_size:
        return num_examples

    rand = np.random.randint(0, num_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class CurrentBuffer:
    """
    The new task buffer which is actually not needed.
    This part is just for our convenience in coding.
    """
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_examples = 0
        self.attributes = ['examples', 'labels', 'task_labels', 'scores', 'img_id']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     task_labels: torch.Tensor, scores: torch.Tensor, img_id: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param scores: tensor example influence
        :param img_id: tensor image id for compute influence in multi epochs
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, task_labels=None, scores=None, img_id=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, task_labels, scores, img_id)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_examples, self.buffer_size)
            self.num_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if scores is not None:
                    self.scores[index] = scores[i].to(self.device)
                if img_id is not None:
                    self.img_id[index] = img_id[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, fsr=False, current_task=0) -> Tuple:

        if size > self.examples.shape[0]:
            size = self.examples.shape[0]
        choice = np.random.choice(min(self.num_examples, self.examples.shape[0]), size=min(self.num_examples, size), replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        return ret_tuple[:2]


    def get_all_data(self, size: int, transform: transforms=None, fsr=False, current_task=0) -> Tuple:
        if size > self.examples.shape[0]:
            size = self.examples.shape[0]
        choice = torch.from_numpy(np.random.choice(min(self.num_examples, self.examples.shape[0]), size=min(self.num_examples, size), replace=False)).to(self.device)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        return ret_tuple + (choice,)


    def get_input_score(self, img_id, shape):
        a = [torch.where(self.img_id == img_id[i])[0].cpu().numpy()[0] for i in range(shape)]
        index = torch.tensor(a)
        return index, self.scores[index]


    def replace_scores(self, index, mem_scores):
        for i in range(len(mem_scores)):
            self.scores[index[i]] = mem_scores[i].to(self.device)

    def ourkmeans(self, replace):
        num_centers = replace
        kmeansdata = torch.reshape(self.examples, [self.examples.shape[0], -1])
        centers, codes, distance = cluster(kmeansdata, num_centers, self.device)
        return codes, distance


    def score(self, replace, codes):
        ranking = []
        for i in range(replace):
            kmeams_label = torch.where(codes == i)
            maxscore_index = kmeams_label[0][torch.argmin(self.scores[kmeams_label][:, 2]).item()].item()
            ranking.append(maxscore_index)
        ranking = torch.tensor(ranking).to(self.device)
        return ranking