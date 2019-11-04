class Avg:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, i):
        self.sum += i
        self.count += 1

    def avg(self):
        return self.sum / self.count