import time
import collections
import numpy

import conlleval

class SequenceLabelingEvaluator(object):
    def __init__(self, main_label_id, label2id=None, conll_eval=False):
        self.main_label_id = main_label_id
        self.label2id = label2id
        self.conll_eval = conll_eval

        self.cost_sum = 0.0
        self.correct_sum = 0.0
        self.main_predicted_count = 0
        self.main_total_count = 0
        self.main_correct_count = 0
        self.token_count = 0
        self.start_time = time.time()

        if self.label2id is not None:
            self.id2label = collections.OrderedDict()
            for label in self.label2id:
                self.id2label[self.label2id[label]] = label

        self.conll_format = []

    def append_data(self, cost, predicted_labels, word_ids, label_ids):
        self.cost_sum += cost
        self.token_count += label_ids.size
        self.correct_sum += numpy.equal(predicted_labels, label_ids).sum()
        self.main_predicted_count += (predicted_labels == self.main_label_id).sum()
        self.main_total_count += (label_ids == self.main_label_id).sum()
        self.main_correct_count += ((predicted_labels == self.main_label_id)*(label_ids == self.main_label_id)).sum()

        for i in range(word_ids.shape[0]):
            for j in range(word_ids.shape[1]-2):
                try:
                    self.conll_format.append(str(word_ids[i][j+1]) + "\t" + str(self.id2label[label_ids[i][j]]) + "\t" + str(self.id2label[predicted_labels[i][j]]))
                except KeyError:
                    print("Unexpected label id in predictions.") # Probably means the CRF decided to predict a start/end label, which it shouldn't
            self.conll_format.append("")


    def get_results(self, name):
        p = (float(self.main_correct_count) / float(self.main_predicted_count)) if (self.main_predicted_count > 0) else 0.0
        r = (float(self.main_correct_count) / float(self.main_total_count)) if (self.main_total_count > 0) else 0.0
        f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
        f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (p+r > 0.0) else 0.0

        results = collections.OrderedDict()
        results[name + "_cost_avg"] = self.cost_sum / float(self.token_count)
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_main_predicted_count"] = self.main_predicted_count
        results[name + "_main_total_count"] = self.main_total_count
        results[name + "_main_correct_count"] = self.main_correct_count
        results[name + "_p"] = p
        results[name + "_r"] = r
        results[name + "_f"] = f
        results[name + "_f05"] = f05
        results[name + "_accuracy"] = self.correct_sum / float(self.token_count)
        results[name + "_token_count"] = self.token_count
        results[name + "_time"] = float(time.time()) - float(self.start_time)

        if self.label2id is not None and self.conll_eval == True:
            conll_counts = conlleval.evaluate(self.conll_format)
            conll_metrics_overall, conll_metrics_by_type = conlleval.metrics(conll_counts)
            results[name + "_conll_accuracy"] = float(conll_counts.correct_tags) / float(conll_counts.token_counter)
            results[name + "_conll_p"] = conll_metrics_overall.prec
            results[name + "_conll_r"] = conll_metrics_overall.rec
            results[name + "_conll_f"] = conll_metrics_overall.fscore
#            for i, m in sorted(conll_metrics_by_type.items()):
#                results[name + "_conll_p_" + str(i)] = m.prec
#                results[name + "_conll_r_" + str(i)] = m.rec
#                results[name + "_conll_f_" + str(i)] = m.fscore #str(m.fscore) + " " + str(conll_counts.t_found_guessed[i])

        return results



