import torch
import torch.nn as nn

def set_activation(name):
    if name == 'none':
        return lambda x,y,z:(x,0)
    pre=None
    if '-' in name:
        pre=name.split('-')[0]
        name = name.split('-')[1]

    if name in ['relu', 'ReLU', 'Relu']:
        act = nn.ReLU()
    elif name in ['tanh', 'Tanh']:
        act = nn.Tanh()
    elif name in ['sigmoid', 'Sigmoid']:
        act= nn.Sigmoid()
    else:
        raise ValueError('Unknown activation function: {}'.format(name))

    if pre is None:
        res = lambda x,y,z : (act(x), 0 )
    elif pre in ['My', 'my']:
        res = lambda x,y,z: (ReLU2(x-y,z),y.clone()) if x.size(-1)==y.size(-1) else (act(x),0)
        #((z)*(-z+x-y>0)+act(-z+x-y),y.clone()) if x.size(-1)==y.size(-1) else (act(x),0)
    elif pre in ['Res', 'res']:
        res = lambda x,y,z: (act(x),y.clone()) if x.size(-1)==y.size(-1) else (act(x),0)
    else:
        raise ValueError('Unknown activation function: {}'.format(name))

    return res

def ReLU2(x,c):
    return x*(x>c)

