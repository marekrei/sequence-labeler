import sys
import collections
import numpy
import random
import math
import gc
import os
import re
import ConfigParser
import theano

from sequence_labeler import SequenceLabeler
from sequence_labeling_evaluator import SequenceLabelingEvaluator

floatX=theano.config.floatX

def read_input_files(file_paths):
    sentences = []
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            words, labels = [], []
            for line in f:
                if len(line.strip()) > 0:
                    line_parts = line.strip().split()
                    assert(len(line_parts) >= 2)
                    words.append(line_parts[0])
                    labels.append(line_parts[-1])
                elif len(line.strip()) == 0 and len(words) > 0:
                    sentences.append((words, labels))
                    words, labels = [], []
            if len(words) > 0:
                raise ValueError("The format expects an empty line at the end of the file in: " + file_path)
    return sentences


def create_batches_of_sentence_ids(sentences, max_batch_size):
    sentence_ids_by_length = collections.OrderedDict()
    for i in range(len(sentences)):
        length = len(sentences[i][0])
        if length not in sentence_ids_by_length:
            sentence_ids_by_length[length] = []
        sentence_ids_by_length[length].append(i)

    batches_of_sentence_ids = []
    for sentence_length in sentence_ids_by_length:
        if max_batch_size > 0:
            batch_size = max_batch_size
        else:
            batch_size = int((-1 * max_batch_size) / (sentence_length+2))

        for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
            batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    return batches_of_sentence_ids


def create_feature_matrices_for_batch(sentences, sentence_ids_in_batch, word2id, char2id, label2id, lowercase_words=True, lowercase_chars=False, replace_digits=True, allowed_word_length=-1):
    batch_data = []

    for sentence_id in sentence_ids_in_batch:
        word_ids = map_text_to_ids(" ".join(sentences[sentence_id][0]), word2id, "<s>", "</s>", "<unk>", lowercase=lowercase_words, replace_digits=replace_digits)
        char_ids = [map_text_to_ids("<s>", char2id, "<w>", "</w>", "<cunk>")] + \
                   [map_text_to_ids(" ".join(list(word)), char2id, "<w>", "</w>", "<cunk>", lowercase=lowercase_chars, replace_digits=replace_digits) for word in sentences[sentence_id][0]] + \
                   [map_text_to_ids("</s>", char2id, "<w>", "</w>", "<cunk>")]
        label_ids = map_text_to_ids(" ".join(sentences[sentence_id][1]), label2id)

        assert(len(char_ids) == len(word_ids))
        assert(len(char_ids) == len(label_ids) + 2)

        batch_data.append((word_ids, char_ids, label_ids))

    allowed_word_length = allowed_word_length if allowed_word_length > 0 else 100000000
    max_word_length = min(numpy.array([[len(char_ids) for char_ids in batch_data[i][1]] for i in range(len(batch_data))]).max(), allowed_word_length)
    sentence_length = len(batch_data[0][0])

    word_ids = numpy.zeros((len(sentence_ids_in_batch), sentence_length), dtype=numpy.int32)
    char_ids = numpy.zeros((len(sentence_ids_in_batch), sentence_length, max_word_length), dtype=numpy.int32)
    char_mask = numpy.zeros((len(sentence_ids_in_batch), sentence_length, max_word_length), dtype=numpy.int32)
    label_ids = numpy.zeros((len(sentence_ids_in_batch), sentence_length-2), dtype=numpy.int32)

    for i in range(len(sentence_ids_in_batch)):
        for j in range(sentence_length):
            word_ids[i][j] = batch_data[i][0][j]
        for j in range(sentence_length):
            for k in range(min(max_word_length, len(batch_data[i][1][j]))):
                char_ids[i][j][k] = batch_data[i][1][j][k]
                char_mask[i][j][k] = 1
        for j in range(sentence_length-2):
            label_ids[i][j] = batch_data[i][2][j]

    return word_ids, char_ids, char_mask, label_ids


def process_sentences(sequencelabeler, sentences, testing, learningrate, name, main_label_id, word2id, char2id, label2id, lowercase_words=True, lowercase_chars=False, replace_digits=True, allowed_word_length=-1, max_batch_size=64, conll_eval=False, verbose=True):
    batches_of_sentence_ids = create_batches_of_sentence_ids(sentences, max_batch_size)
    if testing == False:
        random.shuffle(batches_of_sentence_ids)
    evaluator = SequenceLabelingEvaluator(main_label_id, label2id, conll_eval)

    total_sentences = sum([len(x) for x in batches_of_sentence_ids])
    for sentence_ids_in_batch in batches_of_sentence_ids:
        word_ids, char_ids, char_mask, label_ids = create_feature_matrices_for_batch(sentences, sentence_ids_in_batch, word2id, char2id, label2id, lowercase_words, lowercase_chars, replace_digits, allowed_word_length)

        if testing == True:
            cost, predicted_labels, _ = sequencelabeler.test(word_ids, char_ids, char_mask, label_ids)
        else:
            cost, predicted_labels, _ = sequencelabeler.train(word_ids, char_ids, char_mask, label_ids, learningrate)
        evaluator.append_data(cost, predicted_labels, word_ids, label_ids)
        
        word_ids, char_ids, char_mask, label_ids = None, None, None, None
        while gc.collect() > 0:
            pass

    results = evaluator.get_results(name)
    if verbose == True:
        for key in results:
            print key + ": " + str(results[key])
    return results[name + "_cost_sum"], results


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_config(config_section, config_path):
    config_parser = ConfigParser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def generate_word2id_dictionary(texts, min_freq=-1, insert_words=None, lowercase=False, replace_digits=False):
    counter = collections.Counter()
    for text in texts:
        if lowercase:
            text = text.lower()
        if replace_digits:
            text = re.sub(r'\d', '0', text)
        counter.update(text.strip().split())

    word2id = collections.OrderedDict()
    if insert_words is not None:
        for word in insert_words:
            word2id[word] = len(word2id)

    word_count_list = counter.most_common()

    for (word, count) in word_count_list:
        if min_freq <= 0 or count >= min_freq:
            word2id[word] = len(word2id)

    return word2id


def map_text_to_ids(text, word2id, start_token=None, end_token=None, unk_token=None, lowercase=False, replace_digits=False):
    ids = []

    if lowercase:
        text = text.lower()
    if replace_digits:
        text = re.sub(r'\d', '0', text)

    if start_token != None:
        text = start_token + " " + text
    if end_token != None:
        text = text + " " + end_token
    for word in text.strip().split():
        if word in word2id:
            ids.append(word2id[word])
        elif unk_token != None:
            ids.append(word2id[unk_token])
        else:
            raise ValueError("Token not in dictionary and no unknown token assigned: " + word)
    return ids



def preload_vectors(word2id, vector_size, word2vec_path):
    rng = numpy.random.RandomState(123)
    preloaded_vectors = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(len(word2id), vector_size)), dtype=floatX)

    with open(word2vec_path) as f:
        for line in f:
            line_parts = line.strip().split()
            if len(line_parts) <= 2:
                continue
            word = line_parts[0]
            if word in word2id:
                word_id = word2id[word]
                vector = numpy.array(line_parts[1:])
                preloaded_vectors[word_id] = vector
    return preloaded_vectors


def run_experiment(config_path):
    config = parse_config("config", config_path)
    random.seed(config["random_seed"] + 1)
    temp_model_path = config_path + ".model"
    sequencelabeler = None

    # Preparing dictionaries
    if config["path_train"] is not None and len(config["path_train"]) > 0:
        sentences_train = read_input_files(config["path_train"])
        word2id = generate_word2id_dictionary([" ".join(sentence[0]) for sentence in sentences_train], 
                                                        min_freq=config["min_word_freq"], 
                                                        insert_words=["<unk>", "<s>", "</s>"], 
                                                        lowercase=False, 
                                                        replace_digits=True)
        label2id = generate_word2id_dictionary([" ".join(sentence[1]) for sentence in sentences_train])
        char2id = generate_word2id_dictionary([" ".join([" ".join(list(word)) for word in sentence[0]]) for sentence in sentences_train], 
                                                        min_freq=-1, 
                                                        insert_words=["<cunk>", "<w>", "</w>", "<s>", "</s>"], 
                                                        lowercase=False, 
                                                        replace_digits=True)

    if config["load"] is not None and len(config["load"]) > 0:
        sequencelabeler = SequenceLabeler.load(config["load"])
        label2id = sequencelabeler.config["label2id"]
        word2id = sequencelabeler.config["word2id"]
        char2id = sequencelabeler.config["char2id"]

    if config["load"] is None or len(config["load"]) == 0:
        config["n_words"] = len(word2id)
        config["n_chars"] = len(char2id)
        config["n_labels"] = len(label2id)
        config["unk_token"] = "<unk>"
        config["unk_token_id"] = word2id["<unk>"]
        sequencelabeler = SequenceLabeler(config)
        if config['preload_vectors'] is not None:
            new_embeddings = preload_vectors(word2id, config['word_embedding_size'], config['preload_vectors'])
            sequencelabeler.word_embeddings.set_value(new_embeddings)

    if config["path_dev"] is not None and len(config["path_dev"]) > 0:
        sentences_dev = read_input_files(config["path_dev"])

    # printing config
    for key, val in config.items():
            print key, ": ", val
    print "parameter_count: ", sequencelabeler.get_parameter_count()
    print "parameter_count_without_word_embeddings: ", sequencelabeler.get_parameter_count_without_word_embeddings()

    config["word2id"] = word2id
    config["char2id"] = char2id
    config["label2id"] = label2id

    if config["path_train"] is not None and len(config["path_train"]) > 0:
        best_selector_value = 0.0
        learningrate = config["learningrate"]
        for epoch in xrange(config["epochs"]):
            print("EPOCH: " + str(epoch))
            print("learningrate: " + str(learningrate))
            random.shuffle(sentences_train)


            train_cost_sum, results_train = process_sentences(sequencelabeler, sentences_train, testing=False, learningrate=learningrate, name="train", main_label_id=label2id[str(config["main_label"])], word2id=word2id, char2id=char2id, label2id=label2id, lowercase_words=config["lowercase_words"], lowercase_chars=False, replace_digits=config["replace_digits"], allowed_word_length=config["allowed_word_length"], max_batch_size=config['max_batch_size'], conll_eval=config["conll_eval"], verbose=True)

            dev_cost_sum, results_dev = process_sentences(sequencelabeler, sentences_dev, testing=True, learningrate=0.0, name="dev", main_label_id=label2id[str(config["main_label"])], word2id=word2id, char2id=char2id, label2id=label2id, lowercase_words=config["lowercase_words"], lowercase_chars=False, replace_digits=config["replace_digits"], allowed_word_length=config["allowed_word_length"], max_batch_size=config['max_batch_size'], conll_eval=config["conll_eval"], verbose=True)


            if math.isnan(dev_cost_sum) or math.isinf(dev_cost_sum):
                sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
                break

            if (epoch == 0 or (config["best_model_selector"].split(":")[1] == "high" and results_dev[config["best_model_selector"].split(":")[0]] > best_selector_value) 
                           or (config["best_model_selector"].split(":")[1] == "low" and results_dev[config["best_model_selector"].split(":")[0]] < best_selector_value)):
                best_epoch = epoch
                best_selector_value = results_dev[config["best_model_selector"].split(":")[0]]
                sequencelabeler.save(temp_model_path)
            print("best_epoch: " + str(best_epoch))

            while gc.collect() > 0:
                pass

            if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
                break

        # loading the best model so far
        if config["epochs"] > 0:
            sequencelabeler = SequenceLabeler.load(temp_model_path)
            os.remove(temp_model_path)

    if config["save"] is not None and len(config["save"]) > 0:
        sequencelabeler.save(config["save"])

    if config["path_test"] is not None:
        i = 0
        for path_test in config["path_test"].strip().split(":"):
            sentences_test = read_input_files(path_test)
            test_cost_sum, results_test = process_sentences(sequencelabeler, sentences_test, testing=True, learningrate=0.0, name="test"+str(i), main_label_id=label2id[str(config["main_label"])], word2id=word2id, char2id=char2id, label2id=label2id, lowercase_words=config["lowercase_words"], lowercase_chars=False, replace_digits=config["replace_digits"], allowed_word_length=config["allowed_word_length"], max_batch_size=config['max_batch_size'], conll_eval=config["conll_eval"], verbose=True)
            i += 1


if __name__ == "__main__":
    run_experiment(sys.argv[1])
