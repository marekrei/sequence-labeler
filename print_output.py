import sys
import sequence_labeler
import sequence_labeling_experiment
import numpy
import collections
import time

def print_predictions(print_probs, sequencelabeler_model_path, input_file):
    time_loading = time.time()
    model = sequence_labeler.SequenceLabeler.load(sequencelabeler_model_path)

    time_noloading = time.time()
    config = model.config
    predictions_cache = {}

    id2label = collections.OrderedDict()
    for label in config["label2id"]:
        id2label[config["label2id"][label]] = label

    sentences_test = sequence_labeling_experiment.read_input_files(input_file)
    batches_of_sentence_ids = sequence_labeling_experiment.create_batches_of_sentence_ids(sentences_test, config['max_batch_size'])

    for sentence_ids_in_batch in batches_of_sentence_ids:
        word_ids, char_ids, char_mask, label_ids = sequence_labeling_experiment.create_feature_matrices_for_batch(sentences_test, sentence_ids_in_batch, config["word2id"], config["char2id"], config["label2id"], singletons=None, config=config)

        cost, predicted_labels, predicted_probs = model.test_return_probs(word_ids, char_ids, char_mask, label_ids)

        assert(len(sentence_ids_in_batch) == word_ids.shape[0])

        for i in range(len(sentence_ids_in_batch)):
            key = str(sentence_ids_in_batch[i])
            predictions = []
            if print_probs == False:
                for j in range(predicted_labels.shape[1]):
                    predictions.append(id2label[predicted_labels[i][j]])
            elif print_probs == True:
                for j in range(predicted_probs.shape[1]):
                    p_ = ""
                    for k in range(predicted_probs.shape[2]):
                        p_ += str(id2label[k]) + ":" + str(predicted_probs[i][j][k]) + "\t"
                    predictions.append(p_.strip())
            predictions_cache[key] = predictions

    sentence_id = 0
    word_id = 0
    with open(input_file, "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                print("")
                if word_id == 0:
                    continue
                assert(len(predictions_cache[str(sentence_id)]) == word_id), str(len(predictions_cache[str(sentence_id)])) + " " + str(word_id)
                sentence_id += 1
                word_id = 0
                continue
            assert(str(sentence_id) in predictions_cache)
            assert(len(predictions_cache[str(sentence_id)]) > word_id)
            print(line.strip() + "\t" + predictions_cache[str(sentence_id)][word_id].strip())
            word_id += 1
    
    sys.stderr.write("Processed: " + input_file + "\n")
    sys.stderr.write("Elapsed time with loading: " + str(time.time() - time_loading) + "\n")
    sys.stderr.write("Elapsed time without loading: " + str(time.time() - time_noloading) + "\n")




if __name__ == "__main__":
    if sys.argv[1] == "labels":
        print_probs = False
    elif sys.argv[1] == "probs":
        print_probs = True
    else:
        raise ValueError("Unknown value")

    print_predictions(print_probs, sys.argv[2], sys.argv[3])


