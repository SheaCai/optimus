import unittest

from fusion.scheduling import Network

import fusion.nn_models as nns
from fusion.scheduling import batch_size


class TestNNs(unittest.TestCase):
    """
    Tests for NN definitions.
    """

    def test_all_networks(self):
        """
        Get all_networks.
        """
        self.assertIn('alex_net', nns.all_networks())
        self.assertIn('vgg_net', nns.all_networks())
        self.assertGreater(len(nns.all_networks()), 5)

    def test_import_network(self):
        """
        Get import_network.
        """
        batch_size.init(3)
        for name in nns.all_networks():
            network = nns.import_network(name)
            self.assertIsInstance(network, Network)

    def test_import_network_invalid(self):
        """
        Get import_network invalid.
        """
        with self.assertRaisesRegex(ImportError, 'nns: .*defined.*'):
            _ = nns.import_network('aaa')


