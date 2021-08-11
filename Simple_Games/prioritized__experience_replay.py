import numpy as np


class PrioritizedExperienceReplay:

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_inc=0.001):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self.frame = 1
        self.beta = beta
        self.beta_inc = beta_inc
        self.elements_buffer = 0

    def push(self, transition):
        prio = np.max(self.buffer.tree[-self.buffer.capacity:])
        max_prio = prio if prio != 0 else 1.0
        self.buffer.add(max_prio, transition)
        if self.elements_buffer < self.capacity:
            self.elements_buffer += 1

    def sample(self, batch_size):

        batch = []
        priority_total = self.buffer.tree[0]
        priority_range = priority_total / batch_size
        priority_bias = []

        self.beta = np.min([1., self.beta + self.beta_inc])
        indexes = np.empty((batch_size,), dtype=np.int32)

        for i in range(batch_size):
            value = np.random.uniform(priority_range*i, priority_range*(i+1))
            indexes[i], tree_val, data = self.buffer.get(value)
            while data == 0:
                value = np.random.uniform(priority_range * i, priority_range * (i + 1))
                indexes[i], tree_val, data = self.buffer.get(value)
            else:
                priority_bias.append(tree_val)
                batch.append(data)

        bias_weigth = np.power(self.buffer.n_elements * (priority_bias / priority_total), -self.beta)
        bias_weigth = bias_weigth / bias_weigth.max()
        return batch, indexes, bias_weigth

    def update_priorities(self, indexes, priorities_difference):
        for index, priority in zip(indexes, priorities_difference):
            self.buffer.update(index, (priority + 1e-5)**self.alpha)

    def __len__(self):
        return len(self.buffer.tree[self.capacity:])


"""
The following code come from : https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
"""


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_elements = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_elements < self.capacity:
            self.n_elements += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]
