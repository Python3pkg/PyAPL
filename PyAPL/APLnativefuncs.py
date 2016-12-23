
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

# Default behavior for APL is True
ONE_BASED_ARRAYS = True


def gcd(a, b):
    """Compute the greatest common divisor of a and b"""
    while b > 0:
        a, b = b, a % b
    return a


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
    return (float(a) == 0 or float(a) == 1) and (float(w) == 0 or float(w) == 1)


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
    else:  # There's no better way to do this
        if int(a) == 1:
            return np.array([sin(float(w))])
        elif int(a) == 2:
            return np.array([cos(float(w))])
        elif int(a) == 3:
            return np.array([tan(float(w))])
        elif int(a) == 5:
            return np.array([sinh(float(w))])
        elif int(a) == 6:
            return np.array([cosh(float(w))])
        elif int(a) == 7:
            return np.array([tanh(float(w))])
        # Inverse functions
        elif int(a) == -1:
            return np.array([asin(float(w))])
        elif int(a) == -2:
            return np.array([acos(float(w))])
        elif int(a) == -3:
            return np.array([atan(float(w))])
        elif int(a) == -5:
            return np.array([asinh(float(w))])
        elif int(a) == -6:
            return np.array([acosh(float(w))])
        elif int(a) == -7:
            return np.array([atanh(float(w))])


def COMPEQ(a, w):
    '''Check if ⍺ and ⍵ are equal'''
    return np.array([(1 if float(a) == float(w) else 0)])


def COMPNEQ(a, w):
    '''Check if ⍺ and ⍵ are not equal'''
    return np.array([(1 if float(a) != float(w) else 0)])


def COMPLES(a, w):
    '''Check if ⍺ is less than ⍵'''
    return np.array([(1 if float(a) < float(w) else 0)])


def COMPGRE(a, w):
    '''Check if ⍺ is greater than ⍵'''
    return np.array([(1 if float(a) > float(w) else 0)])


def COMPLESQ(a, w):
    '''Check if ⍺ is less than or equal to ⍵'''
    return np.array([(1 if float(a) <= float(w) else 0)])


def COMPGREQ(a, w):
    '''Check if ⍺ is greater than or equal to ⍵'''
    return np.array([(1 if float(a) >= float(w) else 0)])


def BOOLAND(a, w):
    '''Perform a logical AND of the arguments'''
    return np.array([1 if (int(a) == 1 and int(w) == 1) else 0])


def BOOLOR(a, w):
    '''Perform a logical OR of the arguments'''
    return np.array([1 if (int(a) == 1 or int(w) == 1) else 0])


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
        temp = np.ndarray(int(totalNewElements))
        tempravel = w.ravel()
        for i in range(int(totalNewElements)):
            temp[i] = tempravel[i]
        return temp.reshape(list(map(int, list(a))))


def HORROT(a, w):
    '''Horizontally rotate ⍵ ⍺ times'''
    # TODO: Check the shapes of a and w
    flata = False
    if a.shape[0] == 1:
        flata = True
    temp = np.copy(w)
    for index, slice in enumerate(w):
        rotby = int(
            0 - a[0 if flata else index])  # This is negative because APL rotates in the opposite direction as np
        temp[index] = np.roll(slice, rotby)
    return temp


def VERTROT(a, w):
    '''Vertically rotate ⍵ ⍺ times'''
    # TODO: Check the shapes of a and w
    flata = False
    if a.shape[0] == 1:
        flata = True
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
    intermed = []
    if len(w.shape) != 1:
        raise TypeError()
    else:
        for i in range(int(w)):
            intermed.append(i + 1)
    return np.array(intermed)


def SHAPE(w):
    '''Return the shape of ⍵'''
    return np.array([w.shape])


def BOOLNOT(w):
    '''Return the negation of ⍵'''
    if float(w) == 1:
        return np.array([0])
    elif float(w) == 0:
        return np.array([1])


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
    depth = 0
    if not isinstance(w, np.ndarray) and not isinstance(w, list):
        return 0
    if not isinstance(w[0], np.ndarray) and not isinstance(w, list) and w.shape == (1,) and not recursivecall:
        return 0
    for item in list(w):
        current = DEPTH(item, recursivecall=True) + 1
        if current > depth:
            depth = current
    return depth

    # def FIRST(w):
    #     '''Return the first major item of ⍵'''
    #     return np.array()
