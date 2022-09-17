# Exploring Example Influence in Continual Learning
This is the official implementation of the Exploring Example Influence in Continual Learning in PyTorch, published at NeurIPS 2022.

## Requirements
Pytorch>=1.3.0

## Main Algorithm

ours.py is the method to acquire influence and update model by example influence, whose results are reported as 'Ours'.

oursRehSel.py adds the rehearsal selection strategy using example influence, whose results are reported as 'Ours+RehSel'.

min_norm_solvers.py focuses on influence fusion to SP Pareto Optimal.

buffer.py is memory buffer with fixed size.

current_buffer.py is the buffer for new task, which is actually not needed. This part is just for our convenience in coding.

## Cite us

```
@inproceedings{MetaSP,
  title={Exploring Example Influence in Continual Learning},
  author={Sun, Qing and Lyu, Fan and Shang, Fanhua and Feng, Wei and Wan, Liang},
  booktitle={NeurIPS},
  year={2022}
}
```
