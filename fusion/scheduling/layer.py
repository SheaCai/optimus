"""
Layer specification.
"""

from .. import util


class Layer(util.ContentHashClass):
    """
    Base NN layer.
    Includes only the output neuron parameters.
    nofm: # ofmap channels
    hofm, wofm: ofmap height/width
    hstd, wstd: stride height/width
    """

    def __init__(self, nofm, sofm, strd=1, nimg=1):
        if isinstance(sofm, int):
            hofm = sofm
            wofm = sofm
        elif len(sofm) == 2:
            hofm = sofm[0]
            wofm = sofm[1]
        else:
            raise ValueError('Layer: sofm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sofm))

        if isinstance(strd, int):
            hstd = strd
            wstd = strd
        elif isinstance(strd, list) and len(strd) == 2:
            hstd = strd[0]
            wstd = strd[1]
        else:
            raise ValueError('Layer: strd is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(strd))

        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm

        self.hstd = hstd
        self.wstd = wstd
        self.nimg = nimg

    def input_layer(self):
        """
        Get the input layer parameters.
        """
        raise NotImplementedError(self.__class__.__name__)

    @property
    def nifm(self):
        """
        Number of fmap channels of input layer.
        """
        return self.input_layer().nofm

    @property
    def hifm(self):
        """
        Fmap height of input layer.
        """
        return self.input_layer().hofm

    @property
    def wifm(self):
        """
        Fmap width of input layer.
        """
        return self.input_layer().wofm

    def is_valid_padding_sifm(self, sifm):
        """
        Whether the given `sifm` is valid when allowing padding.
        """
        if isinstance(sifm, int):
            hifm = sifm
            wifm = sifm
        elif len(sifm) == 2:
            hifm = sifm[0]
            wifm = sifm[1]
        else:
            raise ValueError('Layer: sifm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sifm))

        h_padding_rng = sorted((self.hofm * self.hstd, self.hifm))
        w_padding_rng = sorted((self.wofm * self.wstd, self.wifm))

        return (h_padding_rng[0] <= hifm <= h_padding_rng[1]
                and w_padding_rng[0] <= wifm <= w_padding_rng[1])

    @property
    def ofmap_size(self):
        """
        Get size of one output fmap.
        """
        return self.hofm * self.wofm * self.nimg

    @property
    def total_ofmap_size(self):
        """
        Get total size of all output fmaps.
        """
        return self.nofm * self.ofmap_size

    @property
    def ifmap_size(self):
        """
        Get size of one input fmap.
        """
        return self.hofm * self.wofm * self.nimg * self.wstd * self.hstd

    @property
    def total_ifmap_size(self):
        """
        Get total size of all input fmaps.
        """
        return self.nifm * self.ifmap_size

    # @property
    # def ifmap_size(self):
    #     """
    #     Get size of one input fmap.
    #     """
    #     return self.input_layer().ofmap_size
    #
    # @property
    # def total_ifmap_size(self):
    #     """
    #     Get total size of all input fmaps.
    #     """
    #     return self.input_layer().total_ofmap_size

    @property
    def ops_per_neuron(self):
        """
        Number of operations per neuron.
        """
        raise NotImplementedError(self.__class__.__name__)

    @property
    def dimension(self):
        raise NotImplementedError(self.__class__.__name__)

    @property
    def total_ops(self):
        """
        Get total number of operations.
        """
        return self.total_ofmap_size * self.ops_per_neuron

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'strd={}'.format(repr((self.hstd, self.wstd)))]))


class InputLayer(Layer):
    """
    NN input layer parameters.
    """

    def input_layer(self):
        return self

    def ops_per_neuron(self):
        return 0


class ConvLayer(Layer):
    """
    NN convolutional layer parameters.
    nifm : # ifmap channels
    nofm : # ofmap channels
    hifm, wifm : ifmap height/width
    hofm, wofm : ofmap height/width
    hfil, wfil : weight filter width/height
    hstd, wstd : stride height/width
    """

    def __init__(self, nifm, nofm, sofm, sfil, strd=1, nimg=1):
        super(ConvLayer, self).__init__(nofm, sofm, strd=strd, nimg=nimg)
        assert self.hstd > 0 and self.wstd > 0
        assert self.hofm > 0 and self.wofm > 0

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('ConvLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil

        hifm = self.hfil + (self.hofm - 1) * self.hstd
        wifm = self.wfil + (self.wofm - 1) * self.wstd
        self.inlayer = Layer(nifm, (hifm, wifm), nimg=nimg)

    def input_layer(self):
        return self.inlayer

    @property
    def ops_per_neuron(self):
        # 2D convolution across all ifmap channels.
        return self.hfil * self.wfil * self.nifm

    @property
    def filter_size(self):
        """
        Get size of one weight filter.
        """
        return self.hfil * self.wfil

    @property
    def total_filter_size(self):
        """
        Get total size of all weight filters.
        """
        return self.nifm * self.nofm * self.filter_size

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.hstd, self.wstd)))]))

    @property
    def dimension(self):
        return [self.wfil, self.hfil, self.nifm, self.wofm, self.hofm, self.nofm, self.nimg]


class FCLayer(ConvLayer):
    """
    NN fully-connected layer parameters.
    As a special case of CONVLayer.
    hifm = hfil, wifm = wfil, strd = 1, hofm = wofm = 1
    """

    def __init__(self, nifm, nofm, sfil=1, nimg=1):
        super(FCLayer, self).__init__(nifm, nofm, 1, sfil, strd=sfil, nimg=nimg)
        assert self.hofm == 1 and self.wofm == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class LocalRegionLayer(Layer):
    """
    NN layer which computes on a local region. The layer has no or limited
    shared weights, whose impact can be ignored during scheduling.
    Includes pooling layer, normalization layer, and element-wise layer.
    """

    def __init__(self, nofm, sofm, nreg, sreg, ntrd=1, strd=1, nimg=1):
        super(LocalRegionLayer, self).__init__(nofm, sofm, strd=strd, nimg=nimg)
        assert self.hstd > 0 and self.wstd > 0
        assert self.hofm > 0 and self.wofm > 0

        if isinstance(sreg, int):
            hreg = sreg
            wreg = sreg
        elif len(sreg) == 2:
            hreg = sreg[0]
            wreg = sreg[1]
        else:
            raise ValueError('LocalRegionLayer: sreg is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sreg))
        if nreg > 1 and (hreg * wreg) > 1:
            raise ValueError('LocalRegionLayer: local region cannot be a mix '
                             'of both n ({}) and h & w ({}, {})'
                             .format(nreg, hreg, wreg))
        self.nreg = nreg
        self.hreg = hreg
        self.wreg = wreg
        self.ntrd = ntrd

        nifm = self.nofm * self.ntrd  # ignore all-zero padding channels.
        hifm = self.hreg + (self.hofm - 1) * self.hstd
        wifm = self.wreg + (self.wofm - 1) * self.wstd
        self.inlayer = Layer(nifm, (hifm, wifm), nimg=nimg)

    def input_layer(self):
        return self.inlayer

    def region_size(self):
        """
        The size of the local region corresponding to one output point.
        """
        return self.nreg * self.hreg * self.wreg

    @property
    def ops_per_neuron(self):
        # Each output point corresponds to merging a local region.
        return self.nreg * self.hreg * self.wreg

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg)),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'ntrd={}'.format(repr(self.ntrd)),
                'strd={}'.format(repr((self.hstd, self.wstd)))]))

    @property
    def dimension(self):
        return [self.wreg, self.hreg, self.nifm, self.wofm, self.hofm, self.nofm, self.nimg]


class DWConvLayer(LocalRegionLayer):
    """
    NN depthwise convolutional layer parameters.
    nifm = nofm
    """
    def __init__(self, nofm, sofm, sreg, strd=1, nimg=1):
        super(DWConvLayer, self).__init__(nofm, sofm, 1, sreg, strd=strd, nimg=nimg)
        assert self.nifm == self.nofm

    @property
    def ops_per_neuron(self):
        # 2D convolution across ifmap plane.
        return self.hreg * self.wreg

    @property
    def filter_size(self):
        """
        Get size of one weight filter.
        """
        return self.hreg * self.wreg

    @property
    def total_filter_size(self):
        """
        Get total size of all weight filters.
        """
        return self.nifm * self.filter_size

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hreg, self.wreg))),
                'strd={}'.format(repr((self.hstd, self.wstd)))]))


class PoolingLayer(LocalRegionLayer):
    """
    NN pooling layer parameters.
    As a special case of LocalRegionLayer.
    nreg = ntrd = 1
    """

    def __init__(self, nofm, sofm, sreg, strd=None, nimg=1):
        if strd is None:
            strd = sreg
        super(PoolingLayer, self).__init__(nofm, sofm, 1, sreg,
                                           ntrd=1, strd=strd, nimg=nimg)
        assert self.nreg == 1
        assert self.ntrd == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'strd={}'.format(repr((self.hstd, self.wstd)))]))


class EltwiseLayer(LocalRegionLayer):
    """
    NN element-wise layer parameters.
    As a special case of LocalRegionLayer.
    nreg = ntrd, sreg = 1
    """

    def __init__(self, nofm, sofm, nreg, nimg=1):
        super(EltwiseLayer, self).__init__(nofm, sofm, nreg, 1,
                                           ntrd=nreg, strd=1, nimg=nimg)
        assert self.hreg == self.wreg == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg))]))


class ConcatLayer(LocalRegionLayer):
    """
    NN concat layer parameters.
    As a special case of LocalRegionLayer.
    nreg = sreg = 1
    """

    def __init__(self, nofm, sofm, nimg=1):
        super(ConcatLayer, self).__init__(nofm, sofm, 1, 1,
                                           ntrd=1, strd=1, nimg=nimg)
        assert self.hreg == self.wreg == self.nreg == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg))]))


class ReorgLayer(LocalRegionLayer):
    """
    NN reorg layer parameters.
    As a special case of LocalRegionLayer.
    nreg = sreg = 1
    """

    def __init__(self, nofm, sofm, sreg, nimg=1):
        super(ReorgLayer, self).__init__(nofm, sofm, 1, sreg,
                                           ntrd=1, strd=sreg, nimg=nimg)
        assert self.nreg == 1 and self.wstd == self.wreg and self.hstd == self.hreg

        nifm = self.nofm * self.wreg * self.hreg  # ignore all-zero padding channels.
        hifm = self.hofm // self.hreg
        wifm = self.wofm // self.wreg
        self.inlayer = Layer(nifm, (hifm, wifm), nimg=nimg)
