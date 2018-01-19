import sys
import labeler
import experiment
import numpy
import collections
import time

def print_predictions(print_probs, model_path, input_file):
    time_loading = time.time()
    model = labeler.SequenceLabeler.load(model_path)

    time_noloading = time.time()
    config = model.config
    predictions_cache = {}

    id2label = collections.OrderedDict()
    for label in model.label2id:
        id2label[model.label2id[label]] = label

    sentences_test = experiment.read_input_files(input_file)
    batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, config["batch_equal_size"], config['max_batch_size'])

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [sentences_test[i] for i in sentence_ids_in_batch]
        cost, predicted_labels, predicted_probs = model.process_batch(batch, is_training=False, learningrate=0.0)

        assert(len(sentence_ids_in_batch) == len(predicted_labels))

        for i in range(len(sentence_ids_in_batch)):
            key = str(sentence_ids_in_batch[i])
            predictions = []
            if print_probs == False:
                for j in range(len(predicted_labels[i])):
                    predictions.append(id2label[predicted_labels[i][j]])
            elif print_probs == True:
                for j in range(len(predicted_probs[i])):
                    p_ = ""
                    for k in range(len(predicted_probs[i][j])):
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


