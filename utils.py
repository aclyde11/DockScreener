from sklearn import metrics

class Avg:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, i):
        self.sum += i
        self.count += 1

    def avg(self):
        return self.sum / self.count

class MetricCollector:
    def __init__(self):
        self.trues = []
        self.preds = []

    def __call__(self, trues, preds):
        trues = trues.cpu().numpy().flatten()
        preds = preds.detach().cpu().numpy().flatten()
        for i in range(trues.shape[0]):
            self.trues.append(trues[i])
            self.preds.append(preds[i])


    def r2(self):
        try:
            return metrics.r2_score(self.trues, self.preds)
        except:
            return -1