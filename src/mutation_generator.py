from keras.models import Sequential
from keras.models import load_model
from keras.layers import SimpleRNN
from keras.layers import deserialize
from keras.layers.convolutional.base_conv import Conv
from keras.layers import Dense
from keras.layers import LSTM
from keras import activations
import tarfile
import shutil
import os


class MutationGenerator:
    def __init__(self, filename):
        if os.path.isdir('workdir'):
            shutil.rmtree('workdir')
        if os.path.isfile('workdir.tar.gz'):
            os.remove('workdir.tar.gz')
        self._tar_workdir = tarfile.open('workdir.tar.gz', 'w:gz')
        self._model = load_model(filename)
        self._all_activations = set()
        self._all_activations.add(activations.relu)
        self._all_activations.add(activations.selu)
        self._all_activations.add(activations.elu)
        self._all_activations.add(activations.exponential)
        self._all_activations.add(activations.hard_sigmoid)
        self._all_activations.add(activations.linear)
        self._all_activations.add(activations.sigmoid)
        self._all_activations.add(activations.softmax)
        self._all_activations.add(activations.softsign)
        self._all_activations.add(activations.softplus)
        self._all_activations.add(activations.tanh)
        # and the list goes on and on...

    def _submit_model(self, v_path):
        self._submit_model_file(self._model)

    def _submit_model_file(self, v_path, model):
        file_name = ('-'.join(v_path)) + 'model.h5'
        model.save(file_name)
        self._tar_workdir.add(file_name)
        os.remove(file_name)

    def close(self):
        self._tar_workdir.close()

    def apply_math_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, Dense) and not isinstance(layer, SimpleRNN):
                continue

            weights = layer.get_weights()

            for neuron_no in range(0, layer.units):
                v_path = ['L%d' % layer_no, 'N%d' % neuron_no, 'MATH_WEIGHT_1']
                weights[0][:, neuron_no] += 1.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_WEIGHT_2')
                weights[0][:, neuron_no] -= 2.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                weights[0][:, neuron_no] += 1.0
                layer.set_weights(weights)  # restore the model
                v_path.pop()
                v_path.append('MATH_WEIGHT_3')
                weights[0][:, neuron_no] *= 2.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_WEIGHT_4')
                weights[0][:, neuron_no] /= 4.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                weights[0][:, neuron_no] *= 2.0
                layer.set_weights(weights)  # restore the model

    def apply_math_weight_conv(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, Conv):
                continue

            weights = layer.get_weights()
            v_path = ['L%d' % layer_no, 'MATH_WEIGHT_CONV_1']
            weights[0] = weights[0] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_WEIGHT_CONV_2')
            weights[0] = weights[0] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            weights[0] = weights[0] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_WEIGHT_CONV_3')
            weights[0] = weights[0] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_WEIGHT_CONV_4')
            weights[0] = weights[0] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            weights[0] = weights[0] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_activation_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, SimpleRNN):
                continue

            weights = layer.get_weights()

            for neuron_no in range(0, layer.units):
                v_path = ['L%d' % layer_no, 'N%d' % neuron_no, 'MATH_ACT_WEIGHT_1']
                weights[1][:, neuron_no] += 1.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_ACT_WEIGHT_2')
                weights[1][:, neuron_no] -= 2.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                weights[1][:, neuron_no] += 1.0
                layer.set_weights(weights)  # restore the model
                v_path.pop()
                v_path.append('MATH_ACT_WEIGHT_3')
                weights[1][:, neuron_no] *= 2.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_ACT_WEIGHT_4')
                weights[1][:, neuron_no] /= 4.0
                layer.set_weights(weights)
                self._submit_model(v_path)
                weights[1][:, neuron_no] *= 2.0
                layer.set_weights(weights)  # restore the model

    def apply_math_lstm_input_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            w_x = weights[0]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_IN_WEIGHT_1']
            w_x[:, :units] = w_x[:, :units] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_2')
            w_x[:, :units] = w_x[:, :units] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, :units] = w_x[:, :units] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_3')
            w_x[:, :units] = w_x[:, :units] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_4')
            w_x[:, :units] = w_x[:, :units] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, :units] = w_x[:, :units] * 2.0
            layer.set_weights(weights)  # restore the model

            w_y = weights[1]

            v_path = ['L%d' % layer_no, 'MATH_LSTM_IN_WEIGHT_5']
            w_y[:, :units] = w_y[:, :units] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_6')
            w_y[:, :units] = w_y[:, :units] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, :units] = w_y[:, :units] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_7')
            w_y[:, :units] = w_y[:, :units] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_WEIGHT_8')
            w_y[:, :units] = w_y[:, :units] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, :units] = w_y[:, :units] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_forget_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            w_x = weights[0]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_FORGET_WEIGHT_1']
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_2')
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_3')
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_4')
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 1:units * 2] = w_x[:, units * 1:units * 2] * 2.0
            layer.set_weights(weights)  # restore the model

            w_y = weights[1]

            v_path = ['L%d' % layer_no, 'MATH_LSTM_FORGET_WEIGHT_5']
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_6')
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_7')
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_WEIGHT_8')
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 1:units * 2] = w_y[:, units * 1:units * 2] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_cell_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            w_x = weights[0]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_CELL_WEIGHT_1']
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_2')
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_3')
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_4')
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 2:units * 3] = w_x[:, units * 2:units * 3] * 2.0
            layer.set_weights(weights)  # restore the model

            w_y = weights[1]

            v_path = ['L%d' % layer_no, 'MATH_LSTM_CELL_WEIGHT_5']
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_6')
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_7')
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_WEIGHT_8')
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 2:units * 3] = w_y[:, units * 2:units * 3] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_output_weight(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            w_x = weights[0]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_OUT_WEIGHT_1']
            w_x[:, units * 3:] = w_x[:, units * 3:] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_2')
            w_x[:, units * 3:] = w_x[:, units * 3:] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 3:] = w_x[:, units * 3:] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_3')
            w_x[:, units * 3:] = w_x[:, units * 3:] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_4')
            w_x[:, units * 3:] = w_x[:, units * 3:] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_x[:, units * 3:] = w_x[:, units * 3:] * 2.0
            layer.set_weights(weights)  # restore the model

            w_y = weights[1]

            v_path = ['L%d' % layer_no, 'MATH_LSTM_OUT_WEIGHT_5']
            w_y[:, units * 3:] = w_y[:, units * 3:] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_6')
            w_y[:, units * 3:] = w_y[:, units * 3:] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 3:] = w_y[:, units * 3:] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_7')
            w_y[:, units * 3:] = w_y[:, units * 3:] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_WEIGHT_8')
            w_y[:, units * 3:] = w_y[:, units * 3:] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            w_y[:, units * 3:] = w_y[:, units * 3:] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_input_bias(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            bias = weights[2]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_IN_BIAS_1']
            bias[:units] = bias[:units] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_BIAS_2')
            bias[:units] = bias[:units] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[:units] = bias[:units] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_IN_BIAS_3')
            bias[:units] = bias[:units] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_IN_BIAS_4')
            bias[:units] = bias[:units] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[:units] = bias[:units] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_forget_bias(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            bias = weights[2]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_FORGET_BIAS_1']
            bias[units * 1:units * 2] = bias[units * 1:units * 2] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_BIAS_2')
            bias[units * 1:units * 2] = bias[units * 1:units * 2] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 1:units * 2] = bias[units * 1:units * 2] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_BIAS_3')
            bias[units * 1:units * 2] = bias[units * 1:units * 2] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_FORGET_BIAS_4')
            bias[units * 1:units * 2] = bias[units * 1:units * 2] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 1:units * 2] = bias[units * 1:units * 2] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_cell_bias(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            bias = weights[2]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_CELL_BIAS_1']
            bias[units * 2:units * 3] = bias[units * 2:units * 3] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_BIAS_2')
            bias[units * 2:units * 3] = bias[units * 2:units * 3] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 2:units * 3] = bias[units * 2:units * 3] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_BIAS_3')
            bias[units * 2:units * 3] = bias[units * 2:units * 3] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_CELL_BIAS_4')
            bias[units * 2:units * 3] = bias[units * 2:units * 3] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 2:units * 3] = bias[units * 2:units * 3] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_lstm_output_bias(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not isinstance(layer, LSTM):
                continue

            units = layer.units
            weights = layer.get_weights()
            bias = weights[2]
            v_path = ['L%d' % layer_no, 'MATH_LSTM_OUT_BIAS_1']
            bias[units * 3:] = bias[units * 3:] + 1.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_BIAS_2')
            bias[units * 3:] = bias[units * 3:] - 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 3:] = bias[units * 3:] + 1.0
            layer.set_weights(weights)  # restore the model
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_BIAS_3')
            bias[units * 3:] = bias[units * 3:] * 2.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_LSTM_OUT_BIAS_4')
            bias[units * 3:] = bias[units * 3:] / 4.0
            layer.set_weights(weights)
            self._submit_model(v_path)
            bias[units * 3:] = bias[units * 3:] * 2.0
            layer.set_weights(weights)  # restore the model

    def apply_math_bias(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not hasattr(layer, 'use_bias') or not layer.use_bias:
                continue

            if not isinstance(layer, Dense) and not isinstance(layer, SimpleRNN):
                continue

            bias_index = 1
            if isinstance(layer, SimpleRNN):
                bias_index = 2

            biases = layer.get_weights()

            for neuron_no in range(0, layer.units):
                v_path = ['L%d' % layer_no, 'N%d' % neuron_no, 'MATH_BIAS_1']
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] + 1.0
                layer.set_weights(biases)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_BIAS_2')
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] - 2.0
                layer.set_weights(biases)
                self._submit_model(v_path)
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] + 1.0
                layer.set_weights(biases)  # restore the model
                v_path.pop()
                v_path.append('MATH_BIAS_3')
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] * 2.0
                layer.set_weights(biases)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_BIAS_4')
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] / 4.0
                layer.set_weights(biases)
                self._submit_model(v_path)
                biases[bias_index][neuron_no] = biases[bias_index][neuron_no] * 2.0
                layer.set_weights(biases)  # restore the model

    def apply_math_bias_conv(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not hasattr(layer, 'use_bias') or not layer.use_bias:
                continue

            if not isinstance(layer, Conv):
                continue

            biases = layer.get_weights()

            v_path = ['L%d' % layer_no, 'MATH_CONV_BIAS_1']
            biases[1] = biases[1] + 1.0
            layer.set_weights(biases)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_CONV_BIAS_2')
            biases[1] = biases[1] - 2.0
            layer.set_weights(biases)
            self._submit_model(v_path)
            biases[1] = biases[1] + 1.0
            layer.set_weights(biases)  # restore the model
            v_path.pop()
            v_path.append('MATH_CONV_BIAS_3')
            biases[1] = biases[1] * 2.0
            layer.set_weights(biases)
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_CONV_BIAS_4')
            biases[1] = biases[1] / 4.0
            layer.set_weights(biases)
            self._submit_model(v_path)
            biases[1] = biases[1] * 2.0
            layer.set_weights(biases)  # restore the model

    def apply_activation_function_replacement(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('activation' in layer.get_config()):
                continue

            v_path = ['L%d' % layer_no, 'ACT_FUNC_REP']
            f = layer.activation
            for g in self._get_replacement(f):
                try:
                    layer.activation = g
                except:
                    layer._activation = g

                v_path.append('%s.%s' % (f.__name__, g.__name__))
                self._submit_model(v_path)
                v_path.pop()

            try:
                layer.activation = f
            except:
                layer._activation = f

    def _get_replacement(self, f):
        result = set(self._all_activations)
        result.remove(f)
        return result

    def apply_math_pool_sz(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('pool_size' in layer.get_config()):
                continue

            old_pool_sz = layer.pool_size

            if len(old_pool_sz) > 1:
                v_path = ['L%d' % layer_no, 'MATH_POOL_SZ_1']
                layer.pool_size = (old_pool_sz[0] + 1, old_pool_sz[1] + 1)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_POOL_SZ_2')
                layer.pool_size = (old_pool_sz[0] - 2, old_pool_sz[1] - 2)
                self._submit_model(v_path)
                layer.pool_size = old_pool_sz  # restore the model

    def apply_math_strides(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('strides' in layer.get_config()):
                continue

            old_strides = layer.strides

            if len(old_strides) > 1:
                v_path = ['L%d' % layer_no, 'MATH_STRIDES_1']
                layer.strides = (old_strides[0] + 1, old_strides[1] + 1)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_STRIDES_2')
                layer.strides = (old_strides[0] - 2, old_strides[1] - 2)
                self._submit_model(v_path)
                layer.strides = old_strides  # restore the model

    def apply_math_kernel_sz(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('kernel_size' in layer.get_config()):
                continue

            old_kernel_sz = layer.kernel_size

            if len(old_kernel_sz) > 1:
                v_path = ['L%d' % layer_no, 'MATH_KERNEL_SZ_1']
                layer.kernel_size = (old_kernel_sz[0] + 1, old_kernel_sz[1] + 1)
                self._submit_model(v_path)
                v_path.pop()
                v_path.append('MATH_KERNEL_SZ_2')
                layer.kernel_size = (old_kernel_sz[0] - 2, old_kernel_sz[1] - 2)
                self._submit_model(v_path)
                layer.kernel_size = old_kernel_sz  # restore the model

    def apply_math_filters(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('filters' in layer.get_config()):
                continue

            old_filters = layer.filters

            v_path = ['L%d' % layer_no, 'MATH_FILTERS_1']
            layer.filters = old_filters + 1
            self._submit_model(v_path)
            v_path.pop()
            v_path.append('MATH_FILTERS_2')
            layer.filters = old_filters - 2
            self._submit_model(v_path)
            layer.filters = old_filters  # restore the model

    def apply_padding_replacement(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('padding' in layer.get_config()):
                continue

            original_padding = layer.padding

            v_path = ['L%d' % layer_no, 'PADDING']
            if original_padding == 'valid':
                layer.padding = 'same'
            else:
                layer.padding = 'valid'
            self._submit_model(v_path)

            layer.padding = original_padding  # restore the model

    def apply_recurrent_activation_function_replacement(self):
        for layer_no in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_no]

            if not ('recurrent_activation' in layer.get_config()):
                continue

            v_path = ['L%d' % layer_no, 'REC_ACT_FUNC_REP']
            f = layer.recurrent_activation
            for g in self._get_replacement(f):
                try:
                    layer.recurrent_activation = g
                except:
                    layer._recurrent_activation = g

                v_path.append('%s.%s' % (f.__name__, g.__name__))
                self._submit_model(v_path)
                v_path.pop()

            try:
                layer.recurrent_activation = f
            except:
                layer._recurrent_activation = f

    def _copy_model(self, ignored_layer_index):
        if not isinstance(self._model.layers[ignored_layer_index], Dense):
            return None
        model = Sequential()
        for layer_index in range(0, len(self._model.layers)):
            if layer_index != 4:
                layer = self._model.layers[layer_index]
                layer_config = layer.get_config()
                cloned_layer = deserialize({'class_name': layer.__class__.__name__, 'config': layer_config})
                model.add(cloned_layer)
                cloned_layer.set_weights(layer.get_weights())
        model.compile(optimizer=self._model.optimizer,
                      loss=self._model.loss,
                      metrics=self._model.metrics)
        return model

    def apply_del_layer(self):
        for layer_index in range(0, len(self._model.layers)):
            cloned_model = self._copy_model(layer_index)

            if cloned_model is not None:
                self._submit_model_file(['L%d' % layer_index, 'DEL_LAYER'], cloned_model)

    def apply_dup_layer(self):
        for layer_index in range(0, len(self._model.layers)):
            layer = self._model.layers[layer_index]

            if not isinstance(layer, Dense):
                continue

            self._model.layers.insert(layer_index + 1, layer)
            self._submit_model(['L%d' % layer_index, 'DUP_LAYER'])
            self._model.layers.pop(layer_index + 1)  # restore the model
