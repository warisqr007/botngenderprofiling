
class BatchHelper:

    def __init__(self,  x, labels, batch_size):
        self.x = x
        # self.x = self.x.reshape(-1, 1)
        self.labels = labels
        self.labels = self.labels.reshape(-1, 1)
        self.batch_size = batch_size

    def next(self, batch_id):
        x_batch = self.x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        labels_batch = self.labels[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        return x_batch, labels_batch
