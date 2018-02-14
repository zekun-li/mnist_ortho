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
                 axis = -1, # along which axis to do the transform
                 ortho_type = 1,# take from two values {1,2} 1: EXP_SYM, 2: INV_SYM
                 use_bias = True,
                 activation=None,
                 decorr_initializer='glorot_uniform',
                 decorr_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 decorr_constraint=None,
                 **kwargs):

        super(Ortho, self).__init__(**kwargs)
        self.axis = axis
        self.ortho_type = ortho_type
        self.use_bias = use_bias
        self.decorr_initializer = initializers.get(decorr_initializer)
        self.decorr_regularizer = regularizers.get(decorr_regularizer)
        self.decorr_constraint = constraints.get(decorr_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)


        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.supports_masking = True


    def build( self, input_shape ) :
        assert len(input_shape) >= 2

        input_dim = input_shape[self.axis] # skew-symmetic matrix has size (input_dim, input_dim)
        n = input_dim
        nb_orth_elems = (n-1)*n / 2

        # vector tensor to hold the elements for constructing skew-symmetric matrix
        self.decorr = self.add_weight(shape = (nb_orth_elems,),
                                     initializer=self.decorr_initializer,
                                     name='decorr' ,
                                     regularizer=self.decorr_regularizer,
                                     constraint=self.decorr_constraint)

        # shape of bias is (w,h,c) for Conv2d and (len_feat,) for Dense
        if self.use_bias:
            self.bias = self.add_weight(shape=input_dim,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.ortho_n = n # dimension of skew-sym matrix
        self.input_spec = InputSpec(ndim = len(input_shape), axes={self.axis: input_dim})
        self.built = True
    
    def _get_orthogonal_matrix_exp(self) :
        # A: skew symmetric matrix
        # O: orthogonal mattrix
        # O = EXP(A)
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
        # matrix exponential
        orth_mat = expm( triu_mat )
        return orth_mat
    
    
    def _get_orthogonal_matrix_inv(self) :
        # A: skew symmetric matrix
        # O: orthogonal mattrix
        # O = (I + A)(I - A)^-1
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
        return orth_mat



    def call(self, inputs):
        if self.ortho_type == 1:
            orth_mat = self._get_orthogonal_matrix_exp()
        else:
            orth_mat = self._get_orthogonal_matrix_inv()

        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        permute_ptn = reduction_axes + [self.axis] # permutation pattern
        permute_back_ptn = list(range(ndim))[:-1] # permutate back
        permute_back_ptn.insert(self.axis, ndim-1)
        # Determines whether permutation is needed.
        needs_permute = (sorted(reduction_axes) != list(range(ndim))[:-1])

        if self.use_bias: # inputs + bias before the orthogonal transform (also before permutation)
            inputs =  K.bias_add(inputs, self.bias)

        if needs_permute:
            output = K.dot(K.permute_dimensions(inputs,permute_ptn), orth_mat)
            output = K.permute_dimensions(output, permute_back_ptn)
        else:
            output = K.dot(inputs, orth_mat)
        
        if self.activation is not None:
            output = self.activation(output)
        return output


    def compute_output_shape(self, input_shape): 
        # output_shape = input_shape
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]
        output_shape = list(input_shape)
        #output_shape[-1] = input_shape[-1]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'axis': self.axis,
            'use_bias': self.use_bias,
            'ortho_type': self.ortho_type,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'decorr_initializer': initializers.serialize( self.decorr_initializer ),
            'decorr_regularizer': regularizers.serialize( self.decorr_regularizer ),
            'decorr_constraint': constraints.serialize(self.decorr_constraint ) 
        }
        base_config = super(Ortho, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

