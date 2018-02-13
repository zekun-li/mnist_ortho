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
class Ortho( Layer ) :
    def __init__(self, 
                 activation=None,
                 decorr_initializer='glorot_uniform',
                 decorr_regularizer=None,
                 activity_regularizer=None,
                 decorr_constraint=None,
                 **kwargs):

        super(Ortho, self).__init__(**kwargs)
        self.decorr_initializer = initializers.get(decorr_initializer)
        self.decorr_regularizer = regularizers.get(decorr_regularizer)
        self.decorr_constraint = constraints.get(decorr_constraint)

        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build( self, input_shape ) :
        assert len(input_shape) >= 2

        input_dim = input_shape[-1]
        if len(input_shape) == 2: # Dense
            n = input_dim
            nb_orth_elems = (n -1) * n /2
        else: # Conv2D # 2dpooling
            n = np.prod(input_shape[1:])
            nb_orth_elems = (n-1)*n / 2

        self.decorr = self.add_weight(shape = (nb_orth_elems,),
                                     initializer=self.decorr_initializer,
                                     name='decorr' ,
                                     regularizer=self.decorr_regularizer,
                                     constraint=self.decorr_constraint)

        self.ortho_n = n
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
    
    def _get_orthogonal_matrix(self) :
        # 1. create upper triangular matrix using self.decorr
        # 2. create lower triangular matrix using -self.decorr
        # 3. add them up and take matrix exponential

        n = self.ortho_n
        num_triu_entries =  (n-1)*n / 2 
        triu_index_matrix = np.zeros([n, n], dtype=np.int32)
        triu_index_matrix[np.triu_indices(n,1)] = np.arange(num_triu_entries)
        triu_index_matrix[np.triu_indices(n,1)[::-1]] = np.arange(num_triu_entries)
        triu_mat = self.decorr[triu_index_matrix] # symmetric matrix with diagonal values be the first element of self.decorr
        triu_mat = tt.extra_ops.fill_diagonal(triu_mat, 0) # set diagonal values zero
        triu_mat = tt.set_subtensor(triu_mat[np.triu_indices(n,1)[::-1]], triu_mat[np.triu_indices(n,1)[::-1]] * -1)

        part1 = tt.identity_like(triu_mat) + triu_mat
        part2 = tt.nlinalg.MatrixInverse()( tt.identity_like(triu_mat) - triu_mat)
        orth_mat = K.dot(part1, part2)
        # matrix exponential
        #orth_mat = expm( triu_mat )
        return orth_mat

    def call(self, inputs):
        orth_mat = self._get_orthogonal_matrix()
        output = K.dot(inputs.reshape((inputs.shape[0],-1)), orth_mat)
        
        if self.activation is not None:
            output = self.activation(output)
        return output.reshape(inputs.shape)


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            #'use_bias': self.use_bias,
            #'bias_initializer': initializers.serialize(self.bias_initializer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'bias_constraint': constraints.serialize(self.bias_constraint),
            'decorr_initializer': initializers.serialize( self.decorr_initializer ),
            'decorr_regularizer': regularizers.serialize( self.decorr_regularizer ),
            'decorr_constraint': constraints.serialize(self.decorr_constraint ) 
        }
        base_config = super(Ortho, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

