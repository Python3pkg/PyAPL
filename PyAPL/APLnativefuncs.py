#  All the native APL functions implemented with numpy
#  NOTE: One implementation that might need to be done is
#  adding a function wrapper for checking types and throwing
#  type errors

from math import exp, log, pi, sin, sinh, cos, cosh, tan, tanh, \
    asin, asinh, acos, acosh, atan, atanh, floor, ceil
from decimal import Decimal
from random import randint
import numpy as np
from functools import reduce
import operator
from fractions import gcd

# Default behavior for APL is True
ONE_BASED_ARRAYS = True


def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b / gcd(a, b)


def totalElements(arr):
    return reduce(operator.mul, arr.shape, 1)


def isscalar(w):
    return w.shape[0] == 1


# Returns: False for unmatched types [2 3 4 = 3 4]
# 1 for scalar scalar [3 + 43]
# 2 for scalar table/vector [3 - 14 23 11] OR [41 12 1 > 4]
# 3 for the same shape table/vector [3 2 3 > 1 4 1]
def typeargs(a, w):
    if not isscalar(a) and not isscalar(w):
        if a.shape != w.shape:
            return -1
        else:
            return 3
    elif isscalar(a) and isscalar(w):
        return 1
    else:
        return 2


def arebool(a, w):
    return (float(a) in [0, 1]) and (float(w) in [0, 1])


# NATIVE APL FUNCTION DEFINITIONS

def PLUS(a, w):
    '''Add two numbers together'''
    return np.array([float(a) + float(w)])


def MINUS(a, w):
    '''Subtract two numbers'''
    return np.array([float(a) - float(w)])


def DIVIDE(a, w):
    '''Divide two numbers.
        Returns error on divide by zero (runtime)'''
    if float(w) == 0:
        raise ZeroDivisionError()
    return np.array([float(a) / float(w)])


def MULTIPLY(a, w):
    '''Multiply numbers together'''
    return np.array([float(a) * float(w)])


def POWER(a, w):
    '''Find ⍺ to the power of ⍵
        Returns complex numbers when expected'''
    return np.array([float(a) ** float(w)])


def LOGBASE(a, w):
    '''Find log base ⍺ of ⍵'''
    return np.array([log(float(w), float(a))])


def RESIDUE(a, w):
    '''Find residue (modulus) of ⍵ with repeated subtraction of ⍺'''
    # Why did I convert to a string then to a decimal? Because computers are dumb
    # http://stackoverflow.com/questions/14763722/python-modulo-on-floats
    return np.array([float(Decimal(str(float(w))) % Decimal(str(float(a))))])


def CEILING(a, w):
    '''Return the greater of ⍺ and ⍵'''
    return w if float(w) > float(a) else a


def FLOOR(a, w):
    '''Return the lesser of ⍺ and ⍵'''
    return a if float(w) > float(a) else w


def CIRCLE(a, w):
    '''Return the corresponding trig function applied'''
    if int(a) not in (1, 2, 3, 5, 6, 7, -1, -2, -3, -5, -6, -7):
        raise NotImplementedError()
    fun = {1: sin, 2: cos, 3: tan, 5: sinh, 6: cosh, 7: tanh,
           -1: asin, -2: acos, -3: atan, -5: asinh, -6: acosh, -7: atanh}[int(a)]
    return np.array([fun(float(w))])


def COMPEQ(a, w):
    '''Check if ⍺ and ⍵ are equal'''
    return np.array([int(float(a) == float(w))])


def COMPNEQ(a, w):
    '''Check if ⍺ and ⍵ are not equal'''
    return np.array([int(float(a) != float(w))])


def COMPLES(a, w):
    '''Check if ⍺ is less than ⍵'''
    return np.array([int(float(a) < float(w))])


def COMPGRE(a, w):
    '''Check if ⍺ is greater than ⍵'''
    return np.array([int(float(a) > float(w))])


def COMPLESQ(a, w):
    '''Check if ⍺ is less than or equal to ⍵'''
    return np.array([int(float(a) <= float(w))])


def COMPGREQ(a, w):
    '''Check if ⍺ is greater than or equal to ⍵'''
    return np.array([int(float(a) >= float(w))])


def BOOLAND(a, w):
    '''Perform a logical AND of the arguments'''
    return np.array([int(int(a) == 1 and int(w) == 1)])


def BOOLOR(a, w):
    '''Perform a logical OR of the arguments'''
    return np.array([int(int(a) == 1 or int(w) == 1)])


def LEASTCM(a, w):
    '''Find the least common multiple of the arguments'''
    return np.array([lcm(int(a), int(w))])


def GREATESTCD(a, w):
    '''Find the greatest common denom of the arguments'''
    return np.array([gcd(int(a), int(w))])


def RESHAPE(a, w):
    '''Reshape ⍵ to be of the shape ⍺'''
    # The total elements that will be in the new array
    totalNewElements = reduce(operator.mul, list(a.ravel()), 1)
    totalCurrentElements = totalElements(w)
    if totalNewElements == totalCurrentElements:
        # The elements are the same, just reshape the array
        return w.reshape(list(map(int, list(a))))
    elif totalNewElements > totalCurrentElements:
        # Default APL behavior. Repeat elements until you reach the new length
        temp = np.ndarray(int(totalNewElements))
        tempravel = w.ravel()
        ntimes = int(totalNewElements / totalCurrentElements)
        for i in range(ntimes):
            for index, item in enumerate(list(tempravel)):
                temp[index + (i * len(tempravel))] = item
        # Now we have the first elements, add the remainder
        remainder = int(totalNewElements - ntimes * len(tempravel))
        for i in range(remainder):
            temp[i + ntimes * len(tempravel)] = tempravel[i]
        return temp.reshape(list(map(int, list(a))))
    else:
        # Default APL behavior. Cut off elements
        tempravel = w.ravel()
        temp = np.array([tempravel[i] for i in range(int(totalNewElements))])
        return temp.reshape(list(map(int, list(a))))


def HORROT(a, w):
    '''Horizontally rotate ⍵ ⍺ times'''
    # TODO: Check the shapes of a and w
    flata = a.shape[0] == 1
    temp = np.copy(w)
    for index, slice in enumerate(w):
        rotby = int(
            0 - a[0 if flata else index])  # This is negative because APL rotates in the opposite direction as np
        temp[index] = np.roll(slice, rotby)
    return temp


def VERTROT(a, w):
    '''Vertically rotate ⍵ ⍺ times'''
    # TODO: Check the shapes of a and w
    flata = a.shape[0] == 1
    # Transpose the array, then transpose it back after rotating
    w = np.transpose(w)
    temp = np.copy(w)
    for index, slice in enumerate(w):
        rotby = int(
            0 - a[0 if flata else index])  # This is negative because APL rotates in the opposite direction as np
        temp[index] = np.roll(slice, rotby)
    return np.transpose(temp)


# MONADIC FUNCTIONS

def INVERT(w):
    '''Return 1 / ⍵'''
    return np.array([1 / float(w)])


def EEXP(w):
    '''Return e to the ⍵'''
    return np.array([exp(float(w))])


def NATLOG(w):
    '''Return the natural log of ⍵'''
    return np.array([log(float(w))])


def ABS(w):
    '''Return the absolute value of ⍵'''
    return np.array([abs(float(w))])


def PITIMES(w):
    '''Return pi times ⍵'''
    return np.array([pi * float(w)])


def COUNT(w):
    '''Return a list of the numbers up to ⍵'''
    if len(w.shape) != 1:
        raise TypeError()
    return np.array([i + 1 for i in range(int(w))])


def SHAPE(w):
    '''Return the shape of ⍵'''
    # Check for leading ones in the shape
    shape = w.shape
    if shape[0] != 1:
        return np.array(shape)
    else:
        # Trim the leading one
        return np.array(shape[1:])


def BOOLNOT(w):
    '''Return the negation of ⍵'''
    return np.array([1 - float(w)])


def TRANSPOSE(w):
    '''Transposes ⍵'''
    return w.transpose()


def VERTFLIP(w):
    '''Flip ⍵ vertically'''
    if len(w.shape) == 1:
        return w[::-1]
    return np.flipud(w)


def HORFLIP(w):
    '''Flip ⍵ horizontally'''
    if len(w.shape) == 1:
        return w[::-1]
    return np.fliplr(w)


def ROUNDUP(w):
    '''Round ⍵ up'''
    return np.array([ceil(float(w))])


def ROUNDDOWN(w):
    '''Round ⍵ down'''
    return np.array([floor(float(w))])


def RANDOM(w):
    '''Return a random number between 0 and ⍵'''
    return np.array([randint(0, int(w))])


def ENCLOSE(w):
    '''Return an enclosed version of ⍵'''
    return np.array([list(w)])


def DEPTH(w, recursivecall=False):
    '''Return how deeply nested something is
        Will return negatives if is of non-uniform depth'''
    # TODO: Check for non-uniform depth and return negative numbers
    if not isinstance(w, np.ndarray) and not isinstance(w, list):
        return 0
    if not isinstance(w[0], np.ndarray) and not isinstance(w, list) and w.shape == (1,) and not recursivecall:
        return 0
    return max([DEPTH(item, recursivecall=True) + 1 for item in list(w)]) if list(w) else 0

    # def FIRST(w):
    #     '''Return the first major item of ⍵'''
    #     return np.array()


def DECODE(a, w):
    '''Return an evaluated polynomial'''
    if a.shape != (1,) or len(w.shape) != 1:
        raise ValueError('Decode can only be called with (scalar)⊥(vector)')
    return sum(float(number) * (float(a) ** (w.shape[0] - index - 1)) for index, number in enumerate(w))


def ENCODE(a, w):
    '''Return the encoded representation of ⍵'''
    if len(a.shape) != 1 or len(w.shape) != 1:
        raise ValueError('Encode only works with (vector/scalar)⊤(vector)')
    number = int(w.copy())
    scalara = a.shape == (1,)
    ret = []
    for base in a:
        if base == 1:
            ret.append([0.0])
        else:
            # pow is the highest power of the base that will be less than the number
            pow = int(floor(log(number, base))) if not number == 0 else 0
            # Now, append digits to ret
            ret.append([floor((number % (base ** (p + 1))) / base ** p) for p in reversed(range(pow + 1))])
    # Pad it with zeroes in order to make it a proper matrix
    longest = max(ret, key=len) if ret else 0
    newret = [[0] * (len(longest) - len(row)) + row for row in ret]
    # This is to fix an issue with double-lists
    return np.array(newret[0] if scalara else newret)


def FIND(a, w):
    '''Return the first index of ⍵ in ⍺'''
    # TODO: Deal with nesting/dimensions
    for index, item in enumerate(a):
        if abs(w - item) < .001:  # Weird closeness function (arbitrary)
            return np.array([index + (ONE_BASED_ARRAYS * 1)])
    return -1
