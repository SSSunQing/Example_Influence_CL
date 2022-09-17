# Exploring Example Influence in Continual Learning
These five python files are the main content of our algorithm. Do not distribute.

ours.py is the method to acquire influence and update model by example influence, whose results are reported as 'Ours'.

oursRehSel.py adds the rehearsal selection strategy using example influence, whose results are reported as 'Ours+RehSel'.

min_norm_solvers.py focuses on influence fusion to SP Pareto Optimal.

buffer.py is memory buffer with fixed size.

current_buffer.py is the buffer for new task, which is actually not needed. This part is just for our convenience in coding.
