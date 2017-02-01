import begin
import logging

# Change the below line to level=logging.DEBUG for more logging info
logging.basicConfig(filename='src.log', level=logging.FATAL)
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

def GENERALAND(a,w): return (BOOLAND if arebool(a, w) else LEASTCM   )(a, w)
def GENERALOR (a,w): return (BOOLOR  if arebool(a, w) else GREATESTCD)(a, w)
def RIGHT     (a,w): return w
def LEFT      (a,w): return a
def ID        (w  ): return w

def subapplydi(func, a, w):
    fun = {'+':PLUS      , '-':MINUS     , '÷':DIVIDE  , '×':MULTIPLY,
           '*':POWER     , '⍟':LOGBASE   , '|':RESIDUE , '⌈':CEILING ,
           '⌊':FLOOR     , '○':CIRCLE    , '=':COMPEQ  , '≠':COMPNEQ ,
           '<':COMPLES   , '>':COMPGRE   , '≥':COMPGREQ, '≤':COMPLESQ,
           '^':GENERALAND, '∨':GENERALAND, '⍴':RESHAPE , '⌽': HORROT ,
           '⊖': VERTROT  , '⊢': RIGHT    , '⊣': LEFT
           }.get(func)
           
    if fun in None:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()
    return fun (a,w)

def applyuserfunc(func, a=None, w=None):
    func = func[1:-1]  # Trim the brackets off the function
    return apl(func, funcargs=[w, a] if a is not None else [w])

def applydi(func, a, w):
    # TODO implement all of the built in functions
    logging.info(('applydi: ' + str(func) + ' ' + str(a) + ' ' + str(w)).encode('utf-8'))
    if len(func) > 1:
        # This is a user function
        return applyuserfunc(func, a, w)

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
            tempscal =  float(w if first else a)
            applied = [ subapplydi(func,
                                    np.array([scalar   if first else tempscal]),
                                    np.array([tempscal if first else scalar]))
                        for scalar in list(templist.flat)] # Applies the function to each member individually
            return np.array(map(float,applied)).reshape(templist.shape)
        elif arg == 3:
            shape = a.shape
            a = a.ravel()
            w = w.ravel()
            applied = [subapplydi(func, np.array([float(a.flat[i])]), np.array([float(w.flat[i])]))
                        for i in range(a.shape[0])]  # a.shape should be equal to w.shape
            return np.array(map(float,applied)).reshape(shape)

    elif func in mixedFuncs:
        return subapplydi(func, a, w)

def subapplymo(func, w):
    fun = {'÷':INVERT ,'*':EEXP   ,'⍟':NATLOG   ,'|':ABS      ,'○' :PITIMES ,
           '⍳':COUNT  ,'⍴':SHAPE  ,'~':BOOLNOT  ,'⍉':TRANSPOSE,'⊖' :VERTFLIP,
           '⌽':HORFLIP,'⌈':ROUNDUP,'⌊':ROUNDDOWN,'?':RANDOM   ,'⊢⊣':ID}.get(func)
    if fun is None:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()
    return fun(w)

def applymo(func, w):
    logging.info(('applymo: ' + str(func) + ' ' + str(w)).encode('utf-8'))
    if len(func) > 1:
        # User function
        return applyuserfunc(func, w=w)
    if func in scalarFuncs or func in '~?':
        if w.shape == 0:
            return subapplymo(func, w)
        else:
            applied = [subapplymo(func, np.array(scalar)) for scalar in list(w.flat)]
            return np.array(map(float,applied))

    elif func in mixedFuncs:
        return subapplymo(func, w)  # Just send the entire thing to the function

def adverb(adv, func, w):
    # For lined arguments, transpose the array then at the end transpose it back
    ret = np.copy(w)  # Do not modify in place
    ret = ret if adv in '⌿⍀' else ret.transpose()

    if adv in '⌿/':
        for index, item in enumerate(ret):
            if index == 0:
                rtot = item if isinstance(item, np.ndarray) else np.array([item])
            else:
                rtot = applydi(func, rtot, item if isinstance(item, np.ndarray) else np.array([item]))
        ret = rtot
    elif adv in '⍀\\':
        # This is where APL does the 'partial sums' of items
        for index, item in enumerate(ret):
            if index == 0:
                rtot = item
            else:
                rtot = applydi(func, rtot, item)
                ret[index] = rtot

    return ret if adv in '⌿⍀' else ret.transpose()

aplnamespace = {}

def apl(string, funcargs=[]):
    out = []
    for str in string.split('\n'): # if '\n' not in string, split will return a 1 element list containing string, so there is no need for separate case
        lex = APLex.APLexer()
        lex.build()
        # logging.info('Parsing string... len = ' + str(len(string)))
        # APL is a right to left language, so we will reverse the token order     
        tokens = lex.inp(str)[::-1]
        # logging.info('Parsing tokens... len = ' + str(len(tokens)))
        a = apl_wrapped(*((tokens, funcargs) if funcargs else (tokens,)))
        if a is not None:
            out.append(a)
    return out[0] if len(out) == 1 else out

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
        logging.info(('tk : ' + str(token.type) + '   ' + str(token.value)).encode('utf-8'))

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
            dimens = re.findall(r'[\[;][^;]+', token.value)
            # Trim the string to remove the semicolons / brackets
            results = [dim[1:-1] if ']' in dim else dim[1:] for dim in dimens]
            outofbracketdata = list(map(apl_wrapped,results))

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
                        pass  # TODO: Throw error. A monadic function was used diadically
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

                if token.type in ['PRIMFUNC','FUNLIT','ASSIGN']:
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

                if token.type in ['NUMBERLIT','VECTORLIT']:
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
                            print('UHOH')  # TODO: Throw error. A monadic function was used diadically
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
