class ElementMutationExecInfo:
    def __init__(self, mutation_exec_results, generated_mutants_total_count, f2p_total, p2f_total):
        self.__mutation_exec_results = mutation_exec_results
        self.__generated_mutants_total_count = generated_mutants_total_count
        self.__f2p_total = f2p_total
        self.__p2f_total = p2f_total

    def get_mutation_exec_results(self):
        return self.__mutation_exec_results

    def get_generated_mutants_total_count(self):
        return self.__generated_mutants_total_count

    def get_f2p_total_count(self):
        return self.__f2p_total

    def get_p2f_total_count(self):
        return self.__p2f_total
