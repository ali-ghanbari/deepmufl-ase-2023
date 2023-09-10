import math


class Metallaxis:
    def __init__(self, mutation_testing_results, failing_tests_total_count):
        self.__mutation_testing_results__ = mutation_testing_results
        self.__tot_f = failing_tests_total_count
        self.__avg_sbi_scores = None
        self.__avg_ochiai_scores = None
        self.__max_sbi_scores = None
        self.__max_ochiai_scores = None

    def calculate_type1_scores(self):
        self.__avg_sbi_scores = dict()
        self.__avg_ochiai_scores = dict()
        self.__max_sbi_scores = dict()
        self.__max_ochiai_scores = dict()
        for (layer_id, layer_mutation_exec_info) in self.__mutation_testing_results__.items():
            max_sbi_score = -math.inf
            max_ochiai_score = -math.inf
            sbi_score_sum = 0
            ochiai_score_sum = 0
            n = 0
            for (n_i_f, n_i_p, _, _) in layer_mutation_exec_info.get_mutation_exec_results():
                if n_i_f + n_i_p == 0 or self.__tot_f == 0:
                    sbi_score = 0
                    ochiai_score = 0
                else:
                    sbi_score = float(n_i_f) / (float(n_i_f + n_i_p))
                    ochiai_score = float(n_i_f) / math.sqrt(float(n_i_f + n_i_p) * float(self.__tot_f))
                max_sbi_score = max(max_sbi_score, sbi_score)
                max_ochiai_score = max(max_ochiai_score, ochiai_score)
                sbi_score_sum = sbi_score_sum + sbi_score
                ochiai_score_sum = ochiai_score_sum + ochiai_score
                n = n + 1
            self.__avg_sbi_scores[layer_id] = float(sbi_score_sum) / float(n)
            self.__avg_ochiai_scores[layer_id] = float(ochiai_score_sum) / float(n)
            self.__max_sbi_scores[layer_id] = max_sbi_score
            self.__max_ochiai_scores[layer_id] = max_ochiai_score

    def calculate_type2_scores(self):
        self.__avg_sbi_scores = dict()
        self.__avg_ochiai_scores = dict()
        self.__max_sbi_scores = dict()
        self.__max_ochiai_scores = dict()
        for (layer_id, layer_mutation_exec_info) in self.__mutation_testing_results__.items():
            max_sbi_score = -math.inf
            max_ochiai_score = -math.inf
            sbi_score_sum = 0
            ochiai_score_sum = 0
            n = 0
            for (_, _, n_i_p, n_i_f) in layer_mutation_exec_info.get_mutation_exec_results():
                if n_i_f + n_i_p == 0 or self.__tot_f == 0:
                    sbi_score = 0
                    ochiai_score = 0
                else:
                    sbi_score = float(n_i_f) / (float(n_i_f + n_i_p))
                    ochiai_score = float(n_i_f) / math.sqrt(float(n_i_f + n_i_p) * float(self.__tot_f))
                max_sbi_score = max(max_sbi_score, sbi_score)
                max_ochiai_score = max(max_ochiai_score, ochiai_score)
                sbi_score_sum = sbi_score_sum + sbi_score
                ochiai_score_sum = ochiai_score_sum + ochiai_score
                n = n + 1
            self.__avg_sbi_scores[layer_id] = float(sbi_score_sum) / float(n)
            self.__avg_ochiai_scores[layer_id] = float(ochiai_score_sum) / float(n)
            self.__max_sbi_scores[layer_id] = max_sbi_score
            self.__max_ochiai_scores[layer_id] = max_ochiai_score

    def get_avg_sbi_scores(self):
        return self.__avg_sbi_scores

    def get_avg_ochiai_scores(self):
        return self.__avg_ochiai_scores

    def get_max_sbi_scores(self):
        return self.__max_sbi_scores

    def get_max_ochiai_scores(self):
        return self.__max_ochiai_scores
