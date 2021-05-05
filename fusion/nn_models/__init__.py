
def import_network(name):
    """
    Import an example network.
    """
    import importlib

    if name not in all_networks():
        raise ImportError('nns: NN {} has not been defined!'.format(name))
    netmod = importlib.import_module('.' + name, 'fusion.nn_models')
    network = netmod.NN
    return network


def all_networks():
    """
    Get all defined networks.
    """
    import os

    nns_dir = os.path.dirname(os.path.abspath(__file__))
    nns = [f[:-len('.py')] for f in os.listdir(nns_dir)
           if f.endswith('.py') and not f.startswith('__')]
    return list(sorted(nns))


