"""
Utilities.
"""
from functools import reduce
from operator import mul
import math


class ContentHashClass(object):
    """
    Class using the content instead of the object ID for hash.
    Such class instance can be used as key in dictionary.
    """

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        if r is NotImplemented:
            return r
        return not r

    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))


def idivc(valx, valy):
    '''
    Integer division and ceiling.

    Return the min integer that is no less than `valx / valy`.
    '''
    if math.isinf(valy):
        if math.isinf(valx):
            return float('nan')
        return 0
    return (valx + valy - 1) // valy


def prod(lst):
    ''' Get the product of a list. '''
    return reduce(mul, lst, 1)


def approx_dividable(total, num, rel_overhead=0.1, abs_overhead=1):
    ''' Whether it is reasonable to divide `total` into `num` parts.
    `rel_overhead` is the allowed max padding overhead measured
    relatively; `abs_overhead` is the allowed max padding
    overhead measured by absolute value.'''
    return total >= num and isclose(
        idivc(total, num) * num, total,
        rel_tol=rel_overhead, abs_tol=abs_overhead)


def factorize(value, num, limits=None):
    '''
    Factorize given `value` into `num` numbers. Return a tuple of length
    `num`.

    Iterate over factor combinations of which the product is `value`.

    `limits` is a (num-1)-length tuple, specifying the upper limits for the
    first num-1 factors.
    '''
    if limits is None:
        limits = [float('inf')] * (num - 1)
    assert len(limits) >= num - 1
    limits = list(limits[:num-1]) + [float('inf')]

    factors = [1] * num
    while True:
        # Calculate the last factor.
        factors[-1] = idivc(value, prod(factors[:-1]))
        if prod(factors) == value \
                and all(f <= l for f, l in zip(factors, limits)):
            yield tuple(factors)

        # Update the first n - 1 factor combination, backwards.
        lvl = num - 1
        while lvl >= 0:
            factors[lvl] += 1
            if prod(factors[:lvl+1]) <= value:
                break
            factors[lvl] = 1
            lvl -= 1
        if lvl < 0:
            return


def closest_factor(value, factor):
    '''
    Return the maximum factor of `value` that is no larger than `factor` (if
    any), and the minimum factor of `value` that is no less than `factor` (if
    any), as a tuple.
    '''
    if not isinstance(value, int):
        raise TypeError('value must be integers.')

    if value < 0 or factor < 0:
        raise ValueError('arguments must not be negative.')

    res = tuple()

    # Maximum no-larger factor.
    if factor >= 1:
        f = int(factor) + 1
        while f > factor:
            f -= 1
        while True:
            if f != 0 and value % f == 0:
                break
            f -= 1
        assert f <= factor and value % f == 0
        res += (f,)

    # Minimum no-smaller factor.
    if factor <= abs(value):
        f = int(factor) - 1
        while f < factor:
            f += 1
        while True:
            if f != 0 and value % f == 0:
                break
            f += 1
        assert f >= factor and value % f == 0
        res += (f,)

    return res


def get_ith_range(rng, idx, num):
    '''
    Divide the full range `rng` into `num` parts, and get the `idx`-th range.
    '''
    length = rng[1] - rng[0]
    beg = rng[0] + idx * length // num
    end = rng[0] + (idx + 1) * length // num
    assert end <= rng[1]
    return beg, end


def gcd(*values):
    '''
    Get the greatest common divisor of the given values.
    '''
    if any(not isinstance(v, int) for v in values):
        raise TypeError('value must be integers.')
    if any(v <= 0 for v in values):
        raise ValueError('arguments must be positive.')

    if not values:
        raise ValueError('must give at least 1 value.')
    if len(values) == 1:
        return values[0]
    if len(values) > 2:
        return reduce(gcd, values)

    a, b = values
    while b:
        a, b = b, a % b
    return a


def lcm(*values):
    '''
    Get the least common multiple of the given values.
    '''
    if any(not isinstance(v, int) for v in values):
        raise TypeError('value must be integers.')
    if any(v <= 0 for v in values):
        raise ValueError('arguments must be positive.')

    if not values:
        raise ValueError('must give at least 1 value.')
    if len(values) == 1:
        return values[0]
    if len(values) > 2:
        return reduce(lcm, values)

    a, b = values
    return a * b // gcd(a, b)


# def mcm(*num):
#     minimum = 1
#     for i in num:
#         minimum = int(i) * int(minimum) / gcd(int(i), int(minimum))
#         return int(minimum)


def isclose(vala, valb, rel_tol=1e-9, abs_tol=0.0):
    '''
    Whether two values are close to each other.

    Identical to math.isclose() in Python 3.5.
    '''
    return abs(vala - valb) <= max(rel_tol * max(abs(vala), abs(valb)), abs_tol)


def assert_float_eq_int(vfloat, vint, message=''):
    '''
    Check the given float value is equal to the given int value. Print the
    optional message if not equal.
    '''
    if abs(vfloat - vint) > 1:
        raise AssertionError(message + ' {} != {}'.format(vfloat, vint))


def apply(func, argv):
    '''
    Similar to python2 built-in apply function.
    '''
    return func(*argv)


