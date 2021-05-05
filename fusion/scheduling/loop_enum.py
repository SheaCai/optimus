"""
Loop enum type.
Loops include filter width (R), filter height (D),
output width (W), output height (H),
output channel (K), input channel (C),
batch (B).
"""
D = 0
R = 1
C = 2
W = 3
H = 4
K = 5
B = 6
NUM = 7

table = {0: 'D',
         1: 'R',
         2: 'C',
         3: 'W',
         4: 'H',
         5: 'K',
         6: 'B'}

loop_table = {'D': 0,
              'R': 1,
              'C': 2,
              'W': 3,
              'H': 4,
              'K': 5,
              'B': 6}
