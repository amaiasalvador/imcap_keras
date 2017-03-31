from keras.layers.recurrent import *
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2

class LSTM_sent(Recurrent):
    """
    LSTM with visual sentinel gate, from the paper:
    Lu et al. Knowing When to Look:
              Adaptive Attention via A Visual Sentinel for Image Captioning
    https://arxiv.org/abs/1612.01887
    Outputs the hidden state and the output of the sentinel gate.
    """
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTM_sent, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        # sentinel gate
        self.W_g = self.init((input_dim, self.output_dim),
                             name='{}_W_g'.format(self.name))
        self.U_g = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_g'.format(self.name))
        self.b_g = K.zeros((self.output_dim,), name='{}_b_g'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o,
                                                        self.W_g]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                        self.U_f,
                                                        self.U_c,
                                                        self.U_o,
                                                        self.U_g]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o,
                                                        self.b_g]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o,
                                  self.W_g, self.U_g, self.b_g]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, train=False):
        if self.consume_less == 'cpu':
            if train and (0 < self.dropout_W < 1):
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_g = time_distributed_dense(x, self.W_g, self.b_g, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o,x_g], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            #note that the products of B_W and the bias addition have already
            # been performed on preprocess_input when consume_less is set to 'cpu'
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim: 4 * self.output_dim]
            x_g = x[:, 4 * self.output_dim:]
        else:
            x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
            x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            x_g = K.dot(x * B_W[4], self.W_g) + self.b_g

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))
        g = self.inner_activation(x_g + K.dot(h_tm1 * B_U[4], self.U_g))

        h = o * self.activation(c)
        s = g * self.activation(c)
        return [h,s], [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(5)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.dropout(ones, self.dropout_W) for _ in range(5)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])
        return constants


    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        #the hidden state and the sentinel have the same shape
        return [output_shape, output_shape]

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return [mask, mask]
        else:
            return [None, None]

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_sepc with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        #we need to reorder the batch position, as the default K.rnn()
        # assumes that the output is a single tensor.

        outputs = K.permute_dimensions(outputs, [0,2,1,3])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        #returns a list where the first element is the hidden state and the second sentinel
        if self.return_sequences:
            return [outputs[0], outputs[1]]
        else:
            return [last_output[0],last_output[1]]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(LSTM_sent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def slice_0(x):
    return x[0]
def slice_1(x):
    return x[1]
def slice_output_shape(input_shape):
    return input_shape[0]

if __name__ == '__main__':
    bs = 15
    seqlen = 10
    in_dim = 100
    lstm_dim = 256
    d_dim = 40

    I = Input(batch_shape=(bs,seqlen,in_dim))
    lstm=LSTM_sent(256,return_sequences=True,stateful=True,
                 dropout_W=0.5,dropout_U=0.5,
                 W_regularizer = l2(1e-6),
                 U_regularizer=l2(1e-6), name='hs')
    hs = lstm(I) # hs contains the hidden state and the sentinel
    dh = Lambda(slice_0,output_shape=slice_output_shape)(hs)
    ds = Lambda(slice_1,output_shape=slice_output_shape)(hs)
    dh = TimeDistributed(Dense(d_dim,activation='softmax'))(dh)
    ds = TimeDistributed(Dense(d_dim,activation='softmax'))(ds)

    m = Model(input = I, output = [dh,ds])
    m.summary()

    m.compile(optimizer='adam',loss=['categorical_crossentropy','categorical_crossentropy'])
