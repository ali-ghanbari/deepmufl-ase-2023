from element_mutation_exec_info import ElementMutationExecInfo
from keras.models import load_model
import numpy as np
import tarfile
import os


class MutationExecutor:
    def __init__(self, test_case_splitter, comparator):
        self.__test_case_splitter = test_case_splitter
        self.__comparator = comparator
        self.__passing_test_inputs = None
        self.__passing_test_outputs = None
        self.__failing_test_inputs = None
        self.__failing_test_old = None
        self.__failing_test_expected = None
        self.__tot_p = None
        self.__tot_f = None
        self.__mutants_total_count = None
        self.__non_viable_mutants_total_count = None
        self.__f2p_inputs = None
        self.__p2f_inputs = None
        self.__select_fraction = 1.0
        self.__debug = False

    def test(self):
        self.__passing_test_inputs = np.asarray(self.__test_case_splitter.get_passing_test_inputs())
        self.__passing_test_outputs = self.__test_case_splitter.get_passing_test_outputs()
        self.__failing_test_inputs = np.asarray(self.__test_case_splitter.get_failing_test_inputs())
        self.__failing_test_old = self.__test_case_splitter.get_failing_test_actual_outputs()
        self.__failing_test_expected = self.__test_case_splitter.get_failing_test_expected_outputs()
        self.__tot_p = len(self.__passing_test_inputs)
        self.__tot_f = len(self.__failing_test_inputs)
        self.__mutants_total_count = 0
        self.__non_viable_mutants_total_count = 0

        # find all the mutants
        total_mutants = 0
        all_compressed_mutants_dict = dict()
        with tarfile.open('workdir.tar.gz', 'r:gz') as arc:
            for compressed_mutant in arc.getmembers():
                file_name = compressed_mutant.name
                if file_name.endswith('.h5'):
                    total_mutants = total_mutants + 1
                    layer_id = file_name[:file_name.index('-')]
                    if layer_id not in all_compressed_mutants_dict:
                        all_compressed_mutants_dict[layer_id] = []
                    all_compressed_mutants_dict[layer_id].append(compressed_mutant)

            # convert the lists into iterators
            for layer_id, compressed_mutants in all_compressed_mutants_dict.items():
                all_compressed_mutants_dict[layer_id] = iter(compressed_mutants)

            # select desired fraction of mutants
            self.__mutants_total_count = int(self.__select_fraction * total_mutants)
            print('Selected %2f%% of the mutants (%d)' % (self.__select_fraction * 100, self.__mutants_total_count))
            remaining_mutants = self.__mutants_total_count
            selected_mutants = dict()
            while remaining_mutants > 0:
                for layer_id, lit in all_compressed_mutants_dict.items():
                    if remaining_mutants == 0:
                        break
                    try:
                        compressed_mutant = next(lit)
                        if layer_id not in selected_mutants:
                            selected_mutants[layer_id] = []
                        selected_mutants[layer_id].append(compressed_mutant)
                        remaining_mutants = remaining_mutants - 1
                    except StopIteration:
                        continue

            # test the selected mutants
            result = dict()
            for layer_id, compressed_mutants in selected_mutants.items():
                if not compressed_mutants:
                    continue
                self.__f2p_inputs = set()
                self.__p2f_inputs = set()
                exec_results = []
                for compressed_mutant in compressed_mutants:
                    mutant_name = compressed_mutant.name
                    try:
                        arc.extract(compressed_mutant)
                        model = load_model(mutant_name)
                        n_f2p, n_p2f, n_i_p, n_i_f = self.__exec_mutant(model)
                        exec_results.append((n_f2p, n_p2f, n_i_p, n_i_f))
                    except BaseException as error:
                        self.__non_viable_mutants_total_count = self.__non_viable_mutants_total_count + 1
                        print('Non-viable mutant ' + mutant_name)
                        print('Exception raised: {}'.format(error))
                        exec_results.append((0, 0, 0, 0))
                    finally:
                        os.remove(mutant_name)
                result[layer_id] = ElementMutationExecInfo(exec_results,
                                                           len(compressed_mutants),
                                                           len(self.__f2p_inputs),
                                                           len(self.__p2f_inputs))
        return result

    def __exec_mutant(self, model):
        n_f2p = 0
        n_p2f = 0
        n_i_p = 0
        n_i_f = 0
        if len(self.__passing_test_inputs) > 0:
            p = model.predict(self.__passing_test_inputs)
            for i in range(0, len(p)):
                actual = p[i]
                expected = self.__passing_test_outputs[i]
                if not self.__comparator.compare(expected, actual):
                    self.__p2f_inputs.add(hash(self.__passing_test_inputs[i].tobytes()))
                    n_p2f = n_p2f + 1
                    n_i_p = n_i_p + 1
        if len(self.__failing_test_inputs) > 0:
            p = model.predict(self.__failing_test_inputs)
            for i in range(0, len(p)):
                actual = p[i]
                expected = self.__failing_test_old[i]
                if not self.__comparator.compare(expected, actual):
                    n_i_f = n_i_f + 1
                expected = self.__failing_test_expected[i]
                if self.__comparator.compare(expected, actual):
                    self.__f2p_inputs.add(hash(self.__failing_test_inputs[i].tobytes()))
                    n_f2p = n_f2p + 1
        if self.__debug:
            print('n_f2p=%d, n_p2f=%d, n_i_p=%d, n_i_f%d' % (n_f2p, n_p2f, n_i_p, n_i_f))
        return n_f2p, n_p2f, n_i_p, n_i_f

    def get_passing_tests_total_count(self):
        return self.__tot_p

    def get_failing_tests_total_count(self):
        return self.__tot_f

    def get_mutants_total_count(self):
        return self.__mutants_total_count

    def get_non_viable_mutants_total_count(self):
        return self.__non_viable_mutants_total_count

    def set_mutant_selection_fraction(self, fraction):
        self.__select_fraction = fraction
