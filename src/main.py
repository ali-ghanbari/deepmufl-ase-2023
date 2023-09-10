import numpy as np
import mutation_generator
import test_case_splitter
import os
import mutation_executor
import metallaxis
import muse
import time
import sys
from comparator import comparator_factory

if __name__ == '__main__':
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        raise ValueError('This program expects 6 or 7 command-line arguments')

    model_file = sys.argv[1]
    fraction = float(sys.argv[2])
    X_test = np.load(sys.argv[3])
    y_test = np.load(sys.argv[4])
    model_kind = sys.argv[5]
    delta = 1e-3
    if len(sys.argv) == 7:
        delta = float(sys.argv[6])

    if fraction != 0.25 and fraction != 0.5 and fraction != 0.75 and fraction != 1:
        raise ValueError('Bad fraction value')

    comparator = comparator_factory(model_kind, delta)

    with open('%d.txt' % int(fraction * 100), 'w') as out_file:
        if not os.path.isfile('./workdir.tar.gz'):
            start = time.time()
            mg = mutation_generator.MutationGenerator(model_file)
            mg.apply_math_weight()
            mg.apply_math_weight_conv()
            mg.apply_math_bias()
            mg.apply_math_bias_conv()
            mg.apply_math_filters()
            mg.apply_math_kernel_sz()
            mg.apply_math_strides()
            mg.apply_math_pool_sz()
            mg.apply_padding_replacement()
            mg.apply_activation_function_replacement()
            mg.apply_del_layer()
            mg.apply_dup_layer()
            mg.apply_math_lstm_input_weight()
            mg.apply_math_lstm_forget_weight()
            mg.apply_math_lstm_cell_weight()
            mg.apply_math_lstm_output_weight()
            mg.apply_math_lstm_input_bias()
            mg.apply_math_lstm_forget_bias()
            mg.apply_math_lstm_cell_bias()
            mg.apply_math_lstm_output_bias()
            mg.apply_recurrent_activation_function_replacement()
            mg.close()
            end = time.time()
            print('Mutation generation took %s seconds' % (end - start))
            out_file.write('Mutation generation took %s seconds\n' % (end - start))
        
        start = time.time()
        s = test_case_splitter.TestCaseSplitter(model_file, X_test, y_test, comparator)
        s.split()
        end = time.time()
        print('Test case splitting took %s seconds' % (end - start))
        out_file.write('Test case splitting took %s seconds\n' % (end - start))

        start = time.time()
        mt = mutation_executor.MutationExecutor(s, comparator)
        mt.set_mutant_selection_fraction(fraction)
        mtr = mt.test()
        end = time.time()
        print('Mutation execution took %s seconds' % (end - start))
        out_file.write('Mutation execution took %s seconds\n' % (end - start))
        
        print('deepmufl generated %d mutants out of which %d turned out to be non-viable'
              % (mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))
        out_file.write('deepmufl generated %d mutants out of which %d turned out to be non-viable\n'
                       % (mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))
        
        me = metallaxis.Metallaxis(mtr, mt.get_failing_tests_total_count())
        
        me.calculate_type1_scores()
        print('Metallaxis - Type 1:')
        print('\tSBI Avg: %s' % me.get_avg_sbi_scores())
        print('\tSBI Max: %s' % me.get_max_sbi_scores())
        print('')
        print('\tOchiai Avg: %s' % me.get_avg_ochiai_scores())
        print('\tOchiai Max: %s' % me.get_max_ochiai_scores())
        out_file.write('Metallaxis - Type 1:\n')
        out_file.write('\tSBI Avg: %s\n' % me.get_avg_sbi_scores())
        out_file.write('\tSBI Max: %s\n' % me.get_max_sbi_scores())
        out_file.write('\n')
        out_file.write('\tOchiai Avg: %s\n' % me.get_avg_ochiai_scores())
        out_file.write('\tOchiai Max: %s\n' % me.get_max_ochiai_scores())
        
        print('--------------------')
        out_file.write('--------------------\n')
        
        me.calculate_type2_scores()
        print('Metallaxis - Type 2:')
        print('\tSBI Avg: %s' % me.get_avg_sbi_scores())
        print('\tSBI Max: %s' % me.get_max_sbi_scores())
        print('')
        print('\tOchiai Avg: %s' % me.get_avg_ochiai_scores())
        print('\tOchiai Max: %s' % me.get_max_ochiai_scores())
        out_file.write('Metallaxis - Type 2:\n')
        out_file.write('\tSBI Avg: %s\n' % me.get_avg_sbi_scores())
        out_file.write('\tSBI Max: %s\n' % me.get_max_sbi_scores())
        out_file.write('\n')
        out_file.write('\tOchiai Avg: %s\n' % me.get_avg_ochiai_scores())
        out_file.write('\tOchiai Max: %s\n' % me.get_max_ochiai_scores())
        
        print('')
        print('====================')
        print('')
        out_file.write('\n')
        out_file.write('====================\n')
        out_file.write('\n')
        
        mu = muse.MUSE(mtr, mt.get_passing_tests_total_count(), mt.get_failing_tests_total_count())
        mu.calculate_scores()
        print('MUSE:')
        print('\t%s' % mu.get_scores())
        out_file.write('MUSE:\n')
        out_file.write('\t%s\n' % mu.get_scores())
