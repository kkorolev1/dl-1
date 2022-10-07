import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
            idx = np.arange(self.num_samples())
            np.random.shuffle(idx)
            self.X = self.X[idx, :]

            if len(self.y.shape) == 2:
                self.y = self.y[idx, :]
            else:
                self.y = self.y[idx]

        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id < len(self):
            if self.batch_id + 1 < len(self):
                batch_slice = slice(self.batch_id * self.batch_size, (self.batch_id + 1) * self.batch_size)
            else:
                batch_slice = slice(self.batch_id * self.batch_size, None)

            X_batch = self.X[batch_slice, :]

            if len(self.y.shape) == 2:
                y_batch = self.y[batch_slice, :]
            else:
                y_batch = self.y[batch_slice]

            self.batch_id += 1
            return (X_batch, y_batch)

        self.batch_id = 0
        raise StopIteration
