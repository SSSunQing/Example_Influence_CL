import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import random
from utils.current_buffer import CurrentBuffer
import higher
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Mem(ContinualModel):
    NAME = 'mem'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Mem, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.transform = None

    def end_task(self, dataset):
        replace = self.args.buffer_size // (self.current_task + 1)
        delete_ind = self.buffer.delete_data_basedscore(replace, task=self.current_task)
        bx, by, b_ids, b_scores, b_imgid, b_ind = self.currentbuffer.get_all_data(self.currentbuffer.num_examples, transform=self.transform)
        new_codes, distance = self.currentbuffer.ourkmeans(replace)
        index = self.currentbuffer.score(replace, new_codes)
        new_score = torch.ones((index.shape[0], 3), device=self.device)
        self.buffer.replace_keshihua_data(delete_ind, bx[index], by[index], b_ids[index], new_score)
        self.current_task  = self.current_task + 1
        self.times = torch.zeros(self.args.buffer_size, device=self.device)
        self.buffer.reset_score()


    def observe(self, inputs, labels, img_id, not_aug_inputs, task, args, epoch):

        real_batch_size = inputs.shape[0]
        task_labels = torch.ones(real_batch_size, dtype=torch.long).to(self.device) * task
        if task == 0:
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()
            score = torch.ones((real_batch_size, 3), dtype=torch.long).to(self.device)
            self.buffer.add_data(examples=inputs, labels=labels, task_labels=task_labels, score=score)
            return loss.item()
        else:
            if epoch<45:
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.opt.step()
                return loss.item()
            else:
                self.opt.zero_grad()
                mem_x, mem_y, mem_ids, mem_score, mem_index = self.buffer.get_data_gmed(self.args.minibatch_size, transform=self.transform, fsr=True, current_task=task)
                total = torch.cat((inputs, mem_x))
                total_labels = torch.cat((labels, mem_y))
                subsample = self.buffer.buffer_size // 10
                bx, by, b_ids, b_score = self.buffer.get_data(subsample, transform=self.transform, fsr=True, current_task=task)
                nx, ny = self.currentbuffer.get_data(subsample, transform=self.transform)
                input_id, get_input_score = self.currentbuffer.get_input_score(img_id, shape=real_batch_size)

                iteration = 1
                with higher.innerloop_ctx(self.net, self.opt) as (meta_model, meta_opt):
                    base1 = torch.ones(total.shape[0], device=self.device)
                    eps1 = torch.zeros(total.shape[0], requires_grad=True, device=self.device)
                    for i in range(iteration):
                        meta_train_outputs = meta_model(total)
                        meta_train_loss = self.loss(meta_train_outputs, total_labels, reduction="none")
                        meta_train_loss = (torch.sum(eps1 * meta_train_loss) + torch.sum(base1 * meta_train_loss)) / torch.tensor(total.shape[0])
                        meta_opt.step(meta_train_loss)
                    meta_val1_outputs = meta_model(bx)
                    meta_val1_loss = self.loss(meta_val1_outputs, by, reduction="mean")
                    eps_grads1 = torch.autograd.grad(meta_val1_loss, eps1)[0].detach()

                with higher.innerloop_ctx(self.net, self.opt) as (meta_model2, meta_opt2):
                    base2 = torch.ones(total.shape[0], device=self.device)
                    eps2 = torch.zeros(total.shape[0], requires_grad=True, device=self.device)
                    for i in range(iteration):
                        meta_train_outputs2 = meta_model2(total)
                        meta_train_loss2 = self.loss(meta_train_outputs2, total_labels, reduction="none")
                        meta_train_loss2 = (torch.sum(eps2 * meta_train_loss2) + torch.sum(base2 * meta_train_loss2)) / torch.tensor(total.shape[0])
                        meta_opt2.step(meta_train_loss2)
                    meta_val2_outputs = meta_model2(nx)
                    meta_val2_loss = self.loss(meta_val2_outputs, ny, reduction="mean")
                    eps_grads2 = torch.autograd.grad(meta_val2_loss, eps2)[0].detach()

                gn = gradient_normalizers([eps_grads1, eps_grads2], [meta_val1_loss.item(), meta_val2_loss.item()], "ours")
                for gr_i in range(len(eps_grads1)):
                    eps_grads1[gr_i] = eps_grads1[gr_i] / gn[0]
                for gr_i in range(len(eps_grads2)):
                    eps_grads2[gr_i] = eps_grads2[gr_i] / gn[1]
                sol, min_norm = MinNormSolver.find_min_norm_element([eps_grads1, eps_grads2])
                w_tilde = sol[0] * eps_grads1 + (1 - sol[0]) * eps_grads2


                # store influence
                mem_score[:, 0] = (mem_score[:, 0] * self.times[mem_index] + eps_grads1[real_batch_size:]) / (self.times[mem_index] + 1)
                mem_score[:, 1] = (mem_score[:, 1] * self.times[mem_index] + eps_grads2[real_batch_size:]) / (self.times[mem_index] + 1)
                mem_score[:, 2] = (mem_score[:, 2] * self.times[mem_index] + w_tilde[real_batch_size:]) / (self.times[mem_index] + 1)
                # update mem score
                self.buffer.replace_score(mem_score, mem_index)
                self.times[mem_index] = self.times[mem_index] + 1

                cur_epoch = epoch - 45
                # store influence
                get_input_score[:, 0] = (get_input_score[:, 0] * cur_epoch + eps_grads1[:real_batch_size]) / (cur_epoch + 1)
                get_input_score[:, 1] = (get_input_score[:, 1] * cur_epoch + eps_grads2[:real_batch_size]) / (cur_epoch + 1)
                get_input_score[:, 2] = (get_input_score[:, 2] * cur_epoch + w_tilde[:real_batch_size]) / (cur_epoch + 1)
                # update new data score
                self.currentbuffer.replace_scores(index=input_id, mem_scores=get_input_score)


                w_tilde = torch.ones(total.shape[0], device=self.device) - 1 * w_tilde
                l1_norm = torch.sum(w_tilde)
                if l1_norm != 0:
                    w = w_tilde / l1_norm
                else:
                    w = w_tilde
                self.opt.zero_grad()
                outputs = self.net(total)
                loss_batch = self.loss(outputs, total_labels, reduction="none")
                loss = torch.sum(w * loss_batch)
                loss.backward()
                self.opt.step()
                return loss.item()