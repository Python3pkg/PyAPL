import begin
import logging

# Change the below line to level=logging.DEBUG for more logging info
logging.basicConfig(filename='src.log', level=logging.WARNING)
from PyAPL import APLex
from PyAPL.APLnativefuncs import *
import numpy as np
import operator
from functools import reduce

from math import exp, log, pi, sin, sinh, cos, cosh, tan, tanh, \
    asin, asinh, acos, acosh, atan, atanh, floor, ceil
from decimal import Decimal
from random import randint

import re

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


# Scalar functions apply their function to each of the parts of a vector
# individually
scalarFuncs = '+ - × ÷ | ⌈ ⌊ * ⍟ ○ ! ^ ∨ ⍲ ⍱ < ≤ = ≥ > ≠'

# Mixed functions do something else
# NOTE that the monadic versions of ~ and ? are scalar, while the diadic versions
# are actually mixed.
mixedFuncs = '⊢ ⊣ ⍴ , ⍪ ⌽ ⊖ ⍉ ↑ ↓ / ⌿ \ ⍀ ⍳ ∊ ⍋ ⍒ ? ⌹ ⊥ ⊤ ⍕ ⍎ ⊂ ⊃ ≡ ⍷ ⌷ ~ ?'

# Adverbs cannot be applied as monads, but can be applied as diads in certain situations
adverbs = r'/\⌿⍀'


def subapplydi(func, a, w):
    if func == '+':
        return PLUS(a, w)
    elif func == '-':
        return MINUS(a, w)
    elif func == '÷':
        return DIVIDE(a, w)
    elif func == '×':
        return MULTIPLY(a, w)
    elif func == '*':
        return POWER(a, w)
    elif func == '⍟':
        return LOGBASE(a, w)
    elif func == '|':
        return RESIDUE(a, w)
    elif func == '⌈':
        return CEILING(a, w)
    elif func == '⌊':
        return FLOOR(a, w)
    elif func == '○':
        return CIRCLE(a, w)
    elif func == '=':
        return COMPEQ(a, w)
    elif func == '≠':
        return COMPNEQ(a, w)
    elif func == '<':
        return COMPLES(a, w)
    elif func == '>':
        return COMPGRE(a, w)
    elif func == '≥':
        return COMPGREQ(a, w)
    elif func == '≤':
        return COMPLESQ(a, w)
    elif func == '^':
        if arebool(a, w):
            return BOOLAND(a, w)
        else:
            return LEASTCM(a, w)
    elif func == '∨':
        if arebool(a, w):
            return BOOLOR(a, w)
        else:
            return GREATESTCD(a, w)
    elif func == '⍴':
        return RESHAPE(a, w)
    elif func == '⌽':
        return HORROT(a, w)
    elif func == '⊖':
        return VERTROT(a, w)
    elif func == '⊢':
        return w
    elif func == '⊣':
        return a
    else:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()


def applyuserfunc(func, a=None, w=None):
    func = func[1:-1]  # Trim the brackets off the function
    return apl(func, funcargs=[w, a] if a is not None else [w])


def applydi(func, a, w):
    # TODO implement all of the built in functions
    logging.info(('applydi: ' + str(func) + ' ' + str(a) + ' ' + str(w)).encode('utf-8'))
    if len(func) > 1:
        # This is a user function
        return applyuserfunc(func, a, w)
    applied = []
    if func in scalarFuncs:
        arg = typeargs(a, w)
        if arg == -1:
            logging.fatal('Mixed lengths used! a = ' + str(a) + ' & w = ' + str(w))
            raise RuntimeError()  # TODO: pretty up error messages
        elif arg == 1:
            return subapplydi(func, a, w)
        elif arg == 2:
            first = False if isscalar(a) else True
            templist = a if first else w
            tempscal = float(a) if not first else float(w)
            for scalar in list(templist.flat):  # Applies the function to each member individually
                applied.append(float(subapplydi(func,
                                                np.array([scalar]) if first else np.array([tempscal]),
                                                np.array([scalar]) if not first else np.array([tempscal]))))
            applied = np.array(applied).reshape(templist.shape)
        elif arg == 3:
            shape = a.shape
            a = a.ravel()
            w = w.ravel()
            for i in range(0, a.shape[0]):  # a.shape should be equal to w.shape
                applied.append(float(subapplydi(func, np.array([float(a.flat[i])]), np.array([float(w.flat[i])]))))
            applied = np.array(applied).reshape(shape)
        return applied

    elif func in mixedFuncs:
        return subapplydi(func, a, w)


def subapplymo(func, w):
    if func == '÷':
        return INVERT(w)
    elif func == '*':
        return EEXP(w)
    elif func == '⍟':
        return NATLOG(w)
    elif func == '|':
        return ABS(w)
    elif func == '○':
        return PITIMES(w)
    elif func == '⍳':
        return COUNT(w)
    elif func == '⍴':
        return SHAPE(w)
    elif func == '~':
        if not arebool(w, w):
            raise TypeError()
        return BOOLNOT(w)
    elif func == '⍉':
        return TRANSPOSE(w)
    elif func == '⊖':
        return VERTFLIP(w)
    elif func == '⌽':
        return HORFLIP(w)
    elif func == '⌈':
        return ROUNDUP(w)
    elif func == '⌊':
        return ROUNDDOWN(w)
    elif func == '?':
        return RANDOM(w)
    elif func in '⊢⊣':
        return w
    else:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()


def applymo(func, w):
    logging.info(('applymo: ' + str(func) + ' ' + str(w)).encode('utf-8'))
    if len(func) > 1:
        # User function
        return applyuserfunc(func, w=w)
    applied = []
    if func in scalarFuncs or func in '~?':
        if w.shape == 0:
            return subapplymo(func, w)
        else:
            for scalar in list(w.flat):
                applied.append(float(subapplymo(func, np.array(scalar))))
            applied = np.array(applied)
            return applied

    elif func in mixedFuncs:
        return subapplymo(func, w)  # Just send the entire thing to the function


def adverb(adv, func, w):
    # For lined arguments, transpose the array then at the end transpose it back
    ret = np.copy(w)  # Do not modify in place
    if adv not in '⌿⍀':
        ret = ret.transpose()

    if adv in '⌿/':
        for index, item in enumerate(ret):
            if index == 0:
                if not isinstance(item, np.ndarray):
                    rtot = np.array([item])
                else:
                    rtot = item
            else:
                if not isinstance(item, np.ndarray):
                    rtot = applydi(func, rtot, np.array([item]))
                else:
                    rtot = applydi(func, rtot, item)
        ret = rtot
    elif adv in '⍀\\':
        # This is where APL does the 'partial sums' of items
        for index, item in enumerate(ret):
            if index == 0:
                rtot = item
            else:
                rtot = applydi(func, rtot, item)
                ret[index] = rtot

    if adv not in '⌿⍀':
        ret = ret.transpose()
    return ret


aplnamespace = {}


def apl(string, funcargs=[]):
    if not '\n' in string:
        lex = APLex.APLexer()
        lex.build()
        # logging.info('Parsing string... len = ' + str(len(string)))
        tokens = lex.inp(string)
        # APL is a right to left language, so we will reverse the token order
        tokens = tokens[::-1]
        # logging.info('Parsing tokens... len = ' + str(len(tokens)))
        if funcargs is []:
            a = apl_wrapped(tokens)
        else:
            a = apl_wrapped(tokens, funcargs)
        return a
    out = []
    for str in string.split('\n'):
        lex = APLex.APLexer()
        lex.build()
        # logging.info('Parsing string... len = ' + str(len(string)))
        tokens = lex.inp(str)
        # APL is a right to left language, so we will reverse the token order
        tokens = tokens[::-1]
        # logging.info('Parsing tokens... len = ' + str(len(tokens)))
        if funcargs is []:
            a = apl_wrapped(tokens)
        else:
            a = apl_wrapped(tokens, funcargs)
        if a is not None:
            out.append(a)
    if len(out) == 1:
        return out[0]
    return out


def bracket(data, index):
    # TODO: Check if the indexes go outside of the data range
    if len(index) > len(data.shape):
        raise RuntimeError()  # rank error
    indexes = []
    for i in index:
        # TODO: Domain error to be thrown here upon unsuccessful conversion
        # Also, allow for examples where one of the dimensional arguments is a vector
        # e.g.: a[1;1 2;1]
        indexes.append(int(i) - 1 if ONE_BASED_ARRAYS else 0)
    # Neat little feature of numpy. It's actually very good with indexing
    return data[tuple(indexes)]


def apl_wrapped(tokens, funcargs=[]):
    ParsingData = None

    opgoto = '→'
    opassign = '←'

    stack = []

    opstack = []

    outofbracketdata = None
    # TODO: Account for scenario like: (2 3 4)[0 2]

    # Return None directly after an assignment
    hideOutp = False

    for token in tokens:

        hideOutp = False

        # logging.info(('tk : ' + str(token.type) + '   ' + str(token.value)).encode('utf-8'))

        if token.type == 'COMMENT':
            continue

        if token.type == 'STATSEP':
            if len(opstack) == 1:  # We have a leftover op
                if opstack[0] != '←':
                    ParsingData = applymo(opstack.pop(), ParsingData)
                else:
                    hideOutp = True
            continue

        if token.type == 'INDEX':
            results = []
            dimens = re.findall(r'[\[;][^;]+', token.value)
            for item in dimens:
                if ']' in item:  # Trim the string to remove the semicolons / brackets
                    item = item[1:-1]
                else:
                    item = item[1:]
                results.append(apl(item))
            outofbracketdata = results

        if token.type == 'RPAREN':
            stack.append((ParsingData, opstack))  # store both this parsing data and the opstack
            ParsingData = None
            opstack = []
            continue

        if token.type == 'LPAREN':

            if len(opstack) == 1:  # e.g.: (÷3+4) - 2
                # Apply the last op as a monad
                ParsingData = applymo(opstack.pop(), ParsingData)

            if stack == []:
                logging.fatal('Unmatched parens. Unable to continue')
                raise RuntimeError()
            else:
                lStack = stack.pop()
                lopstack = lStack[1]
                if len(lopstack) == 1:  # e.g. : (3 + 5)/2
                    ParsingData = applydi(lopstack.pop(), ParsingData, lStack[0])
                else:
                    pass  # Parsing data should stay exactly the same
            continue

        if ParsingData is None:  # For parsing the beginning of new sections, there must be some sort of value

            if token.type == 'NAME':

                if not token.value in aplnamespace:
                    ParsingData = 0
                    logging.error('Referring to unassigned variable : ' + token.value + ' [will assign 0]')
                    aplnamespace[token.value] = 0
                    continue
                else:
                    ParsingData = aplnamespace[token.value]
                    if outofbracketdata is not None:
                        ParsingData = bracket(ParsingData, outofbracketdata)
                        outofbracketdata = None
                    continue

            elif token.type == 'NUMBERLIT' or token.type == 'VECTORLIT':
                ParsingData = token.value
                if outofbracketdata is not None:
                    ParsingData = bracket(ParsingData, outofbracketdata)
                    outofbracketdata = None
                continue
            elif token.type == 'FUNLIT':
                opstack.append(token.value)
            elif token.type == 'ASSIGN':
                # This should only happen when assigning functions
                opstack.append(token.value)
                ParsingData = opstack[0]
            elif token.type == 'FUNCARG':
                if token.value == '⍺':
                    if len(funcargs) != 2:
                        logging.fatal('Attempted to use a diadic function monadically!')
                        raise ValueError('Attempted to use a diadic function monadically')
                    ParsingData = funcargs[1]
                    if outofbracketdata is not None:
                        ParsingData = bracket(ParsingData, outofbracketdata)
                        outofbracketdata = None
                elif token.value == '⍵':
                    if len(funcargs) == 0:
                        raise RuntimeError()  # Shouldn't happen. No funcargs were passed
                    ParsingData = funcargs[0]
                    if outofbracketdata is not None:
                        ParsingData = bracket(ParsingData, outofbracketdata)
                        outofbracketdata = None
        else:
            if len(opstack) == 0:

                if token.type == 'PRIMFUNC':
                    opstack.append(token.value)
                    continue
                elif token.type == 'FUNLIT':
                    opstack.append(token.value)
                    continue
                elif token.type == 'ASSIGN':
                    opstack.append(token.value)
                    continue
                elif token.type == 'NAME':
                    if token.value in aplnamespace:
                        get = aplnamespace[token.value]
                        if not isinstance(get, np.ndarray):
                            opstack.append(get)

            elif len(opstack) >= 1:

                if token.type == 'ASSIGN':
                    opstack.append(token.value)
                    continue

                if token.type == 'NUMBERLIT' or token.type == 'VECTORLIT':
                    if opstack[-1] == opassign:
                        logging.fatal('Attempting to assign a value to a constant, not a symbolic name.')
                        raise RuntimeError()  # TODO: Pretty up error messages
                    # This is the case when it is literal operation value
                    # e.g.: 5 * x
                    ParsingData = applydi(opstack.pop(), token.value, ParsingData)
                    continue

                elif token.type == 'FUNCARG':
                    if token.value == '⍺':
                        if len(funcargs) != 2:
                            logging.fatal('Attempted to use a diadic function monadically!')
                            raise ValueError('Attempted to use a diadic function monadically')
                        else:
                            ParsingData = applydi(opstack.pop(), funcargs[1], ParsingData)
                    elif token.value == '⍵':
                        if len(funcargs) == 0:
                            raise RuntimeError()  # Shouldn't happen. No funcargs were passed
                        ParsingData = applydi(opstack.pop(), funcargs[0], ParsingData)

                elif token.type == 'NAME':
                    if opstack[-1] == opassign:
                        if len(opstack) == 2:
                            # This is the case when we are assigning a function to a value
                            aplnamespace[token.value] = opstack[0]
                            opstack = [opstack[1]]
                            continue
                        # Assign the parsing data to the value in the namespace
                        aplnamespace[token.value] = ParsingData
                        opstack.pop()
                        hideOutp = True
                        continue
                    if not token.value in aplnamespace:
                        logging.error('Referring to unassigned variable : ' + token.value + ' [will use 0]')
                        ParsingData = applydi(opstack.pop(), 0, ParsingData)
                    else:
                        get = aplnamespace[token.value]
                        if isinstance(get, np.ndarray):
                            ParsingData = applydi(opstack.pop(), get, ParsingData)
                        else:
                            # The name must be a function
                            # Apply the previous function as a monad
                            ParsingData = applymo(opstack.pop(), ParsingData)
                            opstack.append(token.value)
                        if outofbracketdata is not None:
                            ParsingData = bracket(ParsingData, outofbracketdata)
                            outofbracketdata = None
                        continue
                elif token.type == 'PRIMFUNC':
                    if opstack[-1] == opassign:
                        # It shouldn't be
                        # Raise syntax error
                        raise RuntimeError()
                    # Check if the thing in the opstack is an adverb
                    if opstack[-1] in adverbs:
                        ParsingData = adverb(opstack.pop(), token.value, ParsingData)
                        continue
                    # Apply the first function as a monadic function then continue parsing
                    ParsingData = applymo(opstack.pop(), ParsingData)
                    opstack.append(token.value)
                    continue

    if len(opstack) == 1:  # We have a leftover op
        if opstack[0] != '←':
            ParsingData = applymo(opstack.pop(), ParsingData)
        else:
            hideOutp = True

    return ParsingData if not hideOutp else None


if __name__ == '__main__':
    while (True):
        a = apl(input('>>>'))
        if a is not None:
            print(a)
