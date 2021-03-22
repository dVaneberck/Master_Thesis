import numpy as np


class PrioritizedExperienceReplay:

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self.frame = 1
        self.beta = beta
        self.old_beta = beta
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

        # upper_nodes = len(self.buffer.tree) - self.capacity
        # priorities = self.buffer.tree[upper_nodes:upper_nodes+self.elements_buffer]
        # probabilities = priorities / priority_total
        # weight_max = (probabilities.min() * self.elements_buffer) ** - self.beta

        indexes = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            value = np.random.uniform(priority_range*i, priority_range*(i+1))
            indexes[i], _, data = self.buffer.get(value)
            batch.append(data)

        # probabilities_total = self.buffer.tree / priority_total
        # weights = ((self.elements_buffer * probabilities_total[indexes]) ** (-self.beta)) / weight_max
        # self.beta_update()

        return batch, indexes

    def update_priorities(self, indexes, priorities_difference):
        # priorities_updated = np.power(priorities_difference, self.alpha)
        for index, priority in zip(indexes, priorities_difference):
            self.buffer.update(index, (priority + 1e-5)**self.alpha)

    # def beta_update(self, descent=1000):
    #     self.frame += 1
    #     self.beta = min(1.0, self.old_beta + self.frame * (1.0 - self.old_beta) / descent)

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

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]
