from sklearn import metrics
import numpy as np

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

    def erf(self, r, y):
        self.preds = np.array(self.preds)
        self.trues = np.array(self.trues)
        indexs_pred = np.argsort(self.preds)
        indexs_true = np.argsort(self.trues)
        return len(
            set(indexs_pred[:int(r * self.preds.shape[0])]).intersection(set(indexs_true[:int(y * self.preds.shape[0])])))

    def erfmax(self, r, y):
        return (int(min(r, y) * self.preds.shape[0]))

    def erftotal(self, r, y):
        return int(min(r,y) * self.preds.shape[0])

    def nefr(self, *i):
        #return self.erf(*i) / self.erfmax(*i) #divides by the max retrival, but I think we should divide by the true possible numebr
        return self.erf(*i) / self.erftotal(*i)

    def r2(self):
        try:
            return metrics.r2_score(self.trues, self.preds)
        except:
            return -1

