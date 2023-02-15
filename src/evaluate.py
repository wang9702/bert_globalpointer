import torch
import numpy as np
from collections import defaultdict
from src.config import ent2id


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()
        self.id2ent = {}
        for k, v in ent2id.items(): 
            self.id2ent[v] = k

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_classification_report(self, y_true, y_pred, digits=2):
        
        true_entities = set(y_true)
        pred_entities = set(y_pred)

        name_width = 0
        d1 = defaultdict(set)
        d2 = defaultdict(set)
        for e in true_entities:
            d1[self.id2ent[e[1]]].add((e[2], e[3]))
            name_width = max(name_width, len(self.id2ent[e[1]]))
        for e in pred_entities:
            d2[self.id2ent[e[1]]].add((e[2], e[3]))

        last_line_heading = 'avg / total'
        width = max(name_width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'

        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

        ps, rs, f1s, s = [], [], [], []
        for type_name, true_entities in d1.items():
            pred_entities = d2[type_name]
            nb_correct = len(true_entities & pred_entities)
            nb_pred = len(pred_entities)
            nb_true = len(true_entities)

            p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
            r = 100 * nb_correct / nb_true if nb_true > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0

            report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            s.append(nb_true)

        report += u'\n'
        # compute averages
        report += row_fmt.format(last_line_heading,
                                np.average(ps, weights=s),
                                np.average(rs, weights=s),
                                np.average(f1s, weights=s),
                                np.sum(s),
                                width=width, digits=digits)
        # print(report)
        return report

    def get_evaluate_fpr(self, y_preds, y_trues):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        pred = []
        true = []
        for y_pred, y_true in zip(y_preds, y_trues):
            # y_pred = y_pred.data.cpu().numpy()
            # y_true = y_true.data.cpu().numpy()
            for b, l, start, end in zip(*torch.where(y_pred > 0)):
                # print(b.item(), l.item(), start.item(), end.item())
                pred.append((b.item(), l.item(), start.item(), end.item()))
            for b, l, start, end in zip(*torch.where(y_true > 0)):
                true.append((b.item(), l.item(), start.item(), end.item()))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y==0 or Z==0:
            return 0, 0, 0
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        report = self.get_evaluate_classification_report(true, pred)
        return f1, precision, recall, report
