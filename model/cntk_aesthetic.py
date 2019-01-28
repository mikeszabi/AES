import numpy as np
import cntk
from cntk import ops, layers
from cntk.contrib.crosstalkcaffe.unimodel.cntkinstance import BlockApiSetup

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)


    imgLow          = cntk.input_variable((227, 227, 3,) , name='imgLow')
    conv1           = convolution(imgLow, strides=(4, 4), auto_padding=[False, False, False], dilation=(1,), groups=1, name='conv1')
    relu1           = layers.Activation(activation = ops.relu, name = 'relu1')(conv1)
    pool1           = pooling(relu1, pooling_type=0, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False], ceil_out_dim=True)
    norm1           = lrn(pool1, k=1, n=3, alpha=9.999999747378752e-05, beta=0.75, name='norm1')
    conv2           = convolution(norm1, strides=(1, 1), auto_padding=[False, True, True], dilation=(1,), groups=2, name='conv2')
    relu2           = layers.Activation(activation = ops.relu, name = 'relu2')(conv2)
    pool2           = pooling(relu2, pooling_type=0, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False], ceil_out_dim=True)
    norm2           = lrn(pool2, k=1, n=3, alpha=9.999999747378752e-05, beta=0.75, name='norm2')
    conv3           = convolution(norm2, strides=(1, 1), auto_padding=[False, True, True], dilation=(1,), groups=1, name='conv3')
    relu3           = layers.Activation(activation = ops.relu, name = 'relu3')(conv3)
    conv4           = convolution(relu3, strides=(1, 1), auto_padding=[False, True, True], dilation=(1,), groups=2, name='conv4')
    relu4           = layers.Activation(activation = ops.relu, name = 'relu4')(conv4)
    conv5           = convolution(relu4, strides=(1, 1), auto_padding=[False, True, True], dilation=(1,), groups=2, name='conv5')
    relu5           = layers.Activation(activation = ops.relu, name = 'relu5')(conv5)
    pool5           = pooling(relu5, pooling_type=0, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False], ceil_out_dim=True)
    fc6_0           = ops.reshape(pool5, (-1,), name = 'fc6_0')
    fc6_1           = dense(fc6_0, name = 'fc6_1')
    relu6           = layers.Activation(activation = ops.relu, name = 'relu6')(fc6_1)
    fc7_0           = ops.reshape(relu6, (-1,), name = 'fc7_0')
    fc7_1           = dense(fc7_0, name = 'fc7_1')
    relu7           = layers.Activation(activation = ops.relu, name = 'relu7')(fc7_1)
    fc8_VividColor_0 = ops.reshape(relu7, (-1,), name = 'fc8_VividColor_0')
    fc8_Light_0     = ops.reshape(relu7, (-1,), name = 'fc8_Light_0')
    fc8_DoF_0       = ops.reshape(relu7, (-1,), name = 'fc8_DoF_0')
    fc8_Symmetry_0  = ops.reshape(relu7, (-1,), name = 'fc8_Symmetry_0')
    fc8_MotionBlur_0 = ops.reshape(relu7, (-1,), name = 'fc8_MotionBlur_0')
    fc8_Repetition_0 = ops.reshape(relu7, (-1,), name = 'fc8_Repetition_0')
    fc8_RuleOfThirds_0 = ops.reshape(relu7, (-1,), name = 'fc8_RuleOfThirds_0')
    fc8_ColorHarmony_0 = ops.reshape(relu7, (-1,), name = 'fc8_ColorHarmony_0')
    fc8_Object_0    = ops.reshape(relu7, (-1,), name = 'fc8_Object_0')
    fc8_Content_0   = ops.reshape(relu7, (-1,), name = 'fc8_Content_0')
    fc8new_0        = ops.reshape(relu7, (-1,), name = 'fc8new_0')
    fc8_BalancingElement_0 = ops.reshape(relu7, (-1,), name = 'fc8_BalancingElement_0')
    fc8_VividColor_1 = dense(fc8_VividColor_0, name = 'fc8_VividColor_1')
    fc8_Light_1     = dense(fc8_Light_0, name = 'fc8_Light_1')
    fc8_DoF_1       = dense(fc8_DoF_0, name = 'fc8_DoF_1')
    fc8_Symmetry_1  = dense(fc8_Symmetry_0, name = 'fc8_Symmetry_1')
    fc8_MotionBlur_1 = dense(fc8_MotionBlur_0, name = 'fc8_MotionBlur_1')
    fc8_Repetition_1 = dense(fc8_Repetition_0, name = 'fc8_Repetition_1')
    fc8_RuleOfThirds_1 = dense(fc8_RuleOfThirds_0, name = 'fc8_RuleOfThirds_1')
    fc8_ColorHarmony_1 = dense(fc8_ColorHarmony_0, name = 'fc8_ColorHarmony_1')
    fc8_Object_1    = dense(fc8_Object_0, name = 'fc8_Object_1')
    fc8_Content_1   = dense(fc8_Content_0, name = 'fc8_Content_1')
    fc8new_1        = dense(fc8new_0, name = 'fc8new_1')
    fc8_BalancingElement_1 = dense(fc8_BalancingElement_0, name = 'fc8_BalancingElement_1')
    relu8_VividColor = layers.Activation(activation = ops.relu, name = 'relu8_VividColor')(fc8_VividColor_1)
    relu8_Light     = layers.Activation(activation = ops.relu, name = 'relu8_Light')(fc8_Light_1)
    relu8_DoF       = layers.Activation(activation = ops.relu, name = 'relu8_DoF')(fc8_DoF_1)
    relu8_Symmetry  = layers.Activation(activation = ops.relu, name = 'relu8_Symmetry')(fc8_Symmetry_1)
    relu8_MotionBlur = layers.Activation(activation = ops.relu, name = 'relu8_MotionBlur')(fc8_MotionBlur_1)
    relu8_Repetition = layers.Activation(activation = ops.relu, name = 'relu8_Repetition')(fc8_Repetition_1)
    relu8_RuleOfThirds = layers.Activation(activation = ops.relu, name = 'relu8_RuleOfThirds')(fc8_RuleOfThirds_1)
    relu8_ColorHarmony = layers.Activation(activation = ops.relu, name = 'relu8_ColorHarmony')(fc8_ColorHarmony_1)
    relu8_Object    = layers.Activation(activation = ops.relu, name = 'relu8_Object')(fc8_Object_1)
    relu8_Content   = layers.Activation(activation = ops.relu, name = 'relu8_Content')(fc8_Content_1)
    relu8new        = layers.Activation(activation = ops.relu, name = 'relu8new')(fc8new_1)
    relu8_BalancingElement = layers.Activation(activation = ops.relu, name = 'relu8_BalancingElement')(fc8_BalancingElement_1)
    fc9_VividColor_0 = ops.reshape(relu8_VividColor, (-1,), name = 'fc9_VividColor_0')
    fc9_Light_0     = ops.reshape(relu8_Light, (-1,), name = 'fc9_Light_0')
    fc9_DoF_0       = ops.reshape(relu8_DoF, (-1,), name = 'fc9_DoF_0')
    fc9_Symmetry_0  = ops.reshape(relu8_Symmetry, (-1,), name = 'fc9_Symmetry_0')
    fc9_MotionBlur_0 = ops.reshape(relu8_MotionBlur, (-1,), name = 'fc9_MotionBlur_0')
    fc9_Repetition_0 = ops.reshape(relu8_Repetition, (-1,), name = 'fc9_Repetition_0')
    fc9_RuleOfThirds_0 = ops.reshape(relu8_RuleOfThirds, (-1,), name = 'fc9_RuleOfThirds_0')
    fc9_ColorHarmony_0 = ops.reshape(relu8_ColorHarmony, (-1,), name = 'fc9_ColorHarmony_0')
    fc9_Object_0    = ops.reshape(relu8_Object, (-1,), name = 'fc9_Object_0')
    fc9_Content_0   = ops.reshape(relu8_Content, (-1,), name = 'fc9_Content_0')
    fc9_BalancingElement_0 = ops.reshape(relu8_BalancingElement, (-1,), name = 'fc9_BalancingElement_0')
    Concat9         = cntk.splice(relu8new, relu8_BalancingElement, relu8_ColorHarmony, relu8_Content, relu8_DoF, relu8_Light, relu8_MotionBlur, relu8_Object, relu8_Repetition, relu8_RuleOfThirds, relu8_Symmetry, relu8_VividColor, axis=2, name='Concat9')
    fc9_VividColor_1 = dense(fc9_VividColor_0, name = 'fc9_VividColor_1')
    fc9_Light_1     = dense(fc9_Light_0, name = 'fc9_Light_1')
    fc9_DoF_1       = dense(fc9_DoF_0, name = 'fc9_DoF_1')
    fc9_Symmetry_1  = dense(fc9_Symmetry_0, name = 'fc9_Symmetry_1')
    fc9_MotionBlur_1 = dense(fc9_MotionBlur_0, name = 'fc9_MotionBlur_1')
    fc9_Repetition_1 = dense(fc9_Repetition_0, name = 'fc9_Repetition_1')
    fc9_RuleOfThirds_1 = dense(fc9_RuleOfThirds_0, name = 'fc9_RuleOfThirds_1')
    fc9_ColorHarmony_1 = dense(fc9_ColorHarmony_0, name = 'fc9_ColorHarmony_1')
    fc9_Object_1    = dense(fc9_Object_0, name = 'fc9_Object_1')
    fc9_Content_1   = dense(fc9_Content_0, name = 'fc9_Content_1')
    fc9_BalancingElement_1 = dense(fc9_BalancingElement_0, name = 'fc9_BalancingElement_1')
    fc10_merge_0    = ops.reshape(Concat9, (-1,), name = 'fc10_merge_0')
    fc10_merge_1    = dense(fc10_merge_0, name = 'fc10_merge_1')
    relu10_merge    = layers.Activation(activation = ops.relu, name = 'relu10_merge')(fc10_merge_1)
    fc11_score_0    = ops.reshape(relu10_merge, (-1,), name = 'fc11_score_0')
    fc11_score_1    = dense(fc11_score_0, name = 'fc11_score_1')
    return fc9_VividColor_1,fc9_Symmetry_1,fc9_Content_1,fc9_RuleOfThirds_1,fc9_ColorHarmony_1,fc9_MotionBlur_1,fc11_score_1,fc9_BalancingElement_1,fc9_Light_1,fc9_DoF_1,fc9_Repetition_1,fc9_Object_1


def pooling(input, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = ops.pooling(input, **kwargs)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer


def convolution(input, name, **kwargs):
    dim = __weights_dict[name]['weights'].ndim

    weight = np.transpose(__weights_dict[name]['weights'], [dim - 1, dim - 2] + list(range(0, dim - 2)))
    w = cntk.Parameter(init=weight, name=name + '_weight')

    input = cntk.transpose(input, [dim - 2] + list(range(0, dim - 2)))

    layer = ops.convolution(w, input, **kwargs)
    if 'bias' in __weights_dict[name]:
        bias = np.reshape(__weights_dict[name]['bias'], [-1] + [1] * (dim - 2))
        b = cntk.Parameter(init=bias, name=name + '_bias')
        layer = layer + b
    layer = cntk.transpose(layer, list(range(1, dim - 1)) + [0])
    return layer


def dense(input, name, **kwargs):
    w = __weights_dict[name]['weights']
    b = __weights_dict[name]['bias'] if 'bias' in __weights_dict[name] else None
    return BlockApiSetup.linear(output_shape=w.shape[1], input_shape=w.shape[0], scale_init=w, bias_init=b, name=name, **kwargs)(input)


def lrn(input, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = BlockApiSetup.lrn(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer

