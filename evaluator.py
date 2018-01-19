import time
import collections
import numpy
import conlleval

class SequenceLabelingEvaluator(object):
    def __init__(self, main_label, label2id, conll_eval=False):
        self.main_label = main_label
        self.label2id = label2id
        self.conll_eval = conll_eval
        self.main_label_id = self.label2id[self.main_label]

        self.cost_sum = 0.0
        self.correct_sum = 0.0
        self.main_predicted_count = 0
        self.main_total_count = 0
        self.main_correct_count = 0
        self.token_count = 0
        self.start_time = time.time()

        self.id2label = collections.OrderedDict()
        for label in self.label2id:
            self.id2label[self.label2id[label]] = label

        self.conll_format = []

    def append_data(self, cost, batch, predicted_labels):
        self.cost_sum += cost
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                token = batch[i][j][0]
                gold_label = batch[i][j][-1]
                predicted_label = self.id2label[predicted_labels[i][j]]

                self.token_count += 1
                if gold_label == predicted_label:
                    self.correct_sum += 1
                if predicted_label == self.main_label:
                    self.main_predicted_count += 1
                if gold_label == self.main_label:
                    self.main_total_count += 1
                if predicted_label == gold_label and gold_label == self.main_label:
                    self.main_correct_count += 1

                self.conll_format.append(token + "\t" + gold_label + "\t" + predicted_label)
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



