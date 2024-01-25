class MUSE:
    def __init__(self, mutation_testing_results, passing_tests_total_count, failing_tests_total_count):
        self.__mutation_testing_results__ = mutation_testing_results
        self.__tot_p = passing_tests_total_count
        self.__tot_f = failing_tests_total_count
        self.__scores = None

    def calculate_scores(self):
        self.__scores = dict()
        for (layer_id, layer_mutation_exec_info) in self.__mutation_testing_results__.items():
            term1 = 0.
            if self.__tot_f > 0:
                term1 = float(layer_mutation_exec_info.get_f2p_total_count()) / float(self.__tot_f)
            term2 = 0.
            if self.__tot_p > 0:
                term2 = float(layer_mutation_exec_info.get_p2f_total_count()) / float(self.__tot_p)
            alpha = term1 * term2
            summation = 0.
            for (n_f2p, n_p2f, _, _) in layer_mutation_exec_info.get_mutation_exec_results():
                term1 = 0.
                if self.__tot_f > 0:
                    term1 = float(n_f2p) / float(self.__tot_f)
                term2 = 0.
                if self.__tot_p > 0:
                    term2 = float(n_p2f) / float(self.__tot_p)
                summation = summation + abs(term1 - alpha * term2)
            n = layer_mutation_exec_info.get_generated_mutants_total_count()
            if n > 0:
                self.__scores[layer_id] = summation / float(n)
            else:
                self.__scores[layer_id] = 0.

    def get_scores(self):
        return self.__scores
