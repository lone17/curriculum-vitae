"""
You are probing your code generation model on a program synthesis benchmark and 1 out of 4 
the candidate solutions produced by your model pass the unit tests of a coding challenge.

What’s the pass@2 metric (in percent) as introduced in the Codex paper (see section 2.1)?

References
- Codex paper: https://arxiv.org/abs/2107.03374    
"""

import numpy as np


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """

    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


print(pass_at_k(4, 1, 2))

"""
\frac{\binom{n-c}{k}}{\binom{n}{k}} 
= \frac{\frac{(n-c)!}{k!(n-c-k)!}}{\frac{n!}{k!(n-k)!}} 
= \frac{(n-c)!}{n!}\frac{(n-k)!}{(n-c-k)!} = \frac{1}{\prod_{i = n-c+1}^{n}i}\prod_{i=n-k-c+1}^{n-k} 
= \frac{(n-k)(n-k-1)...(n-k-c+1)}{n(n-1)...(n-c+1)}
= \frac{n-k}{n}\frac{n-k-1}{n-1}...\frac{n-k-c+1}{n-c+1}
= (1-\frac{k}{n})(1-\frac{k}{n-1})...(1-\frac{k}{n-c+1})
= np.prod(1.0 - k / np.arange(n - c + 1, n + 1)) 
"""
