from keras.layers import Dense
import theano.tensor as tt
from theano.tensor import slinalg 
import numpy as np
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.legacy import interfaces

expm = slinalg.Expm()
class OrthDense( Layer ) :
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 decorr_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 decorr_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 decorr_constraint=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        '''
        super(OrthDense, self).__init__(units,
                                        activation=activation,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        **kwargs)
        '''
        super(OrthDense, self).__init__(**kwargs)
        self.decorr_initializer = initializers.get(decorr_initializer)
        self.decorr_regularizer = regularizers.get(decorr_regularizer)
        self.decorr_constraint = constraints.get(decorr_constraint)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build( self, input_shape ) :
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        nb_orth_elems = (self.units-1)*self.units / 2

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        #self.decorr = []
        #self.decorr = K.zeros(shape = (nb_orth_elems,))
        #for i in range(nb_orth_elems) :
        #    elem_i = self.add_weight(shape=(1,),
        #                             initializer=self.decorr_initializer,
        #                             name='decorr-%d' % i,
        #                             regularizer=self.decorr_regularizer,
        #                             constraint=self.decorr_constraint)
            
        #    self.decorr.append(elem_i)
        #self.decorr = np.array(self.decorr)
        
        self.decorr = self.add_weight(shape = (nb_orth_elems,),
                                     initializer=self.decorr_initializer,
                                     name='decorr' ,
                                     regularizer=self.decorr_regularizer,
                                     constraint=self.decorr_constraint)
        

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
    
    def _get_orthogonal_matrix(self) :
        # 1. create upper triangular matrix using self.decorr
        # 2. create lower triangular matrix using -self.decorr
        # 3. add them up and take matrix exponential
        n = self.units
        num_triu_entries =  (self.units-1)*self.units / 2 
        triu_index_matrix = np.zeros([n, n], dtype=np.int32)
        triu_index_matrix[np.triu_indices(n,1)] = np.arange(num_triu_entries)
        triu_index_matrix[np.triu_indices(n,1)[::-1]] = np.arange(num_triu_entries)
        triu_mat = self.decorr[triu_index_matrix] # symmetric matrix with diagonal values be the first element of self.decorr
        triu_mat = tt.extra_ops.fill_diagonal(triu_mat, 0) # set diagonal values zero
        #triu_mat[np.triu_indices(n,1)[::-1]] = triu_mat[np.triu_indices(n,1)[::-1]] * -1 # skew-symmetric
        triu_mat = tt.set_subtensor(triu_mat[np.triu_indices(n,1)[::-1]], triu_mat[np.triu_indices(n,1)[::-1]] * -1)
        # matrix exponential
        orth_mat = expm( triu_mat )
        return orth_mat
    
    '''
    def _get_orthogonal_matrix(self) :
        # 1. create upper triangular matrix using self.decorr
        # 2. create lower triangular matrix using -self.decorr
        # 3. add them up and take matrix exponential
        n = self.units
        n_triu_entries =  (self.units-1)*self.units / 2 
        r = tt.arange(n)
        tmp_mat = r[np.newaxis, :] + (n_triu_entries - n - (r * (r + 1)) / 2)[::-1, np.newaxis]
        #triu_index_matrix = np.zeros([n, n], dtype=np.int32)
        triu_index_matrix = tt.triu(tmp_mat) + tt.triu(tmp_mat).T - tt.diag(tt.diagonal(tmp_mat))
        sym_matrix = self.decorr[triu_index_matrix]
        orth_mat = sym_matrix
        return orth_mat
    '''
    def call(self, inputs):
        orth_mat = self._get_orthogonal_matrix()
        output_step1 = K.dot(inputs, self.kernel)
        output_step2 = K.dot(output_step1, orth_mat)
        if self.use_bias:
            #output = K.bias_add(output_step2, self.bias)
            output_step2 = K.bias_add(output_step2, self.bias)
        if self.activation is not None:
            #output = self.activation(output_step2)
            output_step2 = self.activation(output_step2)
        return output_step2

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'decorr_initializer': initializers.serialize( self.decorr_initializer ),
            'decorr_regularizer': regularizers.serialize( self.decorr_regularizer ),
            'decorr_constraint': constraints.serialize(self.decorr_constraint ) 
        }
        base_config = super(OrthDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

