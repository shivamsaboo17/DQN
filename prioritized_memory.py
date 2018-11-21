import numpy as np


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        # Number of leaf nodes that will store experiences
        self.capacity = capacity
        # Tree has 2'wice - 1 elements than the leaves
        self.tree = np.zeros(2 * capacity - 1)
        # Store the actual experiences here
        self.data = np.zeros(capacity, dtype=object)
    
    def add(self, priority, data):
        # Adds a data to data and updates corresponding priority
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # Update the sum tree with new priorities
        self.update(tree_index, priority)
        # Increment the data pointer as data is stored in left to right
        # Create a ring buffer with fixed capacity by overwriting new data
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change is the update factor we need to do in sum tree
        change = priority - self.tree[tree_index]
        # Store the priority first in the tree
        self.tree[tree_index] = priority
        # Update the change to parent nodes
        while tree_index !=0 :
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        # Traverse from top of the tree
        parent_index = 0
        while True:
            left_child_index = parent_index * 2 + 1
            right_child_index = left_child_index + 1
            # Base condition to break once tree traversal is complete
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            # Find the node with maximum priority and traverse that subtree
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        # The max priority which we could fetch for given input priority
        data_index = leaf_index - self.capacity + 1
        # Return index of priorities, priorities, and experience
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):

    # HYPERPARAMETERS
    e = 0.01
    a = 0.6
    b = 0.4
    b_del = 0.001
    max_error = 1.0

    def __init__(self, capacity):
        # Build a SumTree
        self.tree = SumTree(capacity)

    def store(self, experience):
        # New experiences are first stored in the tree with max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.max_error
        self.tree.add(max_priority, experience)
    
    def sample(self, n):
        memory_b = []
        b_idx = np.empty((n, ), dtype=np.int32)
        b_ISWeights = np.empty((n, 1), dtype=np.float32)
        # Split the total priority into batch-sized segments
        priority_segment = self.tree.total_priority / n
        # Scheduler for b
        self.b = np.min([1., self.b + self.b_del])
        # Max weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.b)

        for i in range(n):
            # Calculate priority value for each segment to get corresponding experience
            a, b = priority_segment * i, priority_segment * (i+1)
            v = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(v)
            # Fatch and normalize priority
            sampling_probability = priority / self.tree.total_priority
            # Weight update factor depends on sampling probability of the experience and b
            b_ISWeights[i, 0] = np.power(n * sampling_probability, -self.b) / max_weight
            # Store the index and experience
            b_idx[i] = index
            experience = data
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.e
        clipped_errors = np.minimum(abs_errors, self.max_error)
        ps = np.power(clipped_errors, self.a)
        # For the chosen batch, depending on the error, update the priority in sum tree
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
