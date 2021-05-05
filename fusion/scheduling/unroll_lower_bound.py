from .layer import Layer, ConvLayer
from . import loop_enum as le


class _LayerLoopLowerBound(object):
    def __init__(self):
        self.b = self.k = self.h = self.w = self.c = self.r = self.d = 1

    @property
    def size(self):
        return self.b * self.k * self.h * self.w * self.c * self.r * self.d

    @ property
    def dimension(self):
        return [self.d, self.r, self.c, self.w, self.h, self.k, self.b]


class LoopLowerBound(object):

    def __init__(self, loop_lower_bound, unroll_loop, replication_loop=None):
        assert len(loop_lower_bound) == le.NUM

        self.loop_lower_bound_init = loop_lower_bound
        self.unroll_loop = unroll_loop
        self.replication_loop = replication_loop

    @classmethod
    def dataflow(cls, dataflow_info):
        if dataflow_info is not None:
            return cls(dataflow_info['loop_lower_bound'],
                       dataflow_info['unroll_loop'],
                       dataflow_info["replication_loop"])
        else:
            return cls([1, ] * le.NUM, [])

    def __call__(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError('LoopLowerBound: {} is invalid, '
                             'needs to be Layer'.format(layer))

        loop_lower_bound = _LayerLoopLowerBound()
        if isinstance(layer, ConvLayer):
            loop_lower_bound.d = min(layer.wfil, self.loop_lower_bound_init[le.D])
            loop_lower_bound.r = min(layer.hfil, self.loop_lower_bound_init[le.R])
        loop_lower_bound.c = min(layer.nifm, self.loop_lower_bound_init[le.C])
        loop_lower_bound.w = min(layer.wofm, self.loop_lower_bound_init[le.W])
        loop_lower_bound.h = min(layer.hofm, self.loop_lower_bound_init[le.H])
        loop_lower_bound.k = min(layer.nofm, self.loop_lower_bound_init[le.K])
        loop_lower_bound.b = min(layer.nimg, self.loop_lower_bound_init[le.B])

        return loop_lower_bound
