from PyAPL import APLex
from PyAPL.APLnativefuncs import *
import numpy as np
import readline  # For using interactive mode
import re

# Default behavior for APL is True
ONE_BASED_ARRAYS = True

DEBUG_MODE = False

# Scalar functions apply their function to each of the parts of a vector
# individually
scalarFuncs = '+ - × ÷ | ⌈ ⌊ * ⍟ ○ ! ^ ∨ ⍲ ⍱ < ≤ = ≥ > ≠'

# Mixed functions do something else
# NOTE that the monadic versions of ~ and ? are scalar, while the diadic versions
# are actually mixed.
mixedFuncs = '⊢ ⊣ ⍴ , ⍪ ⌽ ⊖ ⍉ ↑ ↓ / ⌿ \ ⍀ ⍳ ∊ ⍋ ⍒ ? ⌹ ⊥ ⊤ ⍕ ⍎ ⊂ ⊃ ≡ ⍷ ⌷ ~ ?'

# Adverbs cannot be applied as monads, but can be applied as diads in certain situations
adverbs = r'/\⌿⍀¨'


def GENERALAND(a, w): return (BOOLAND if arebool(a, w) else LEASTCM)(a, w)


def GENERALOR(a, w): return (BOOLOR if arebool(a, w) else GREATESTCD)(a, w)


def RIGHT(a, w): return w


def LEFT(a, w): return a


def ID(w): return w


def applyuserfunc(func, a=None, w=None):
    func = func[1:-1]  # Trim the brackets off the function
    return apl(func, funcargs=[w, a] if a is not None else [w])


def applydi(func, a, w):
    fun = {'+': PLUS, '-': MINUS, '÷': DIVIDE, '×': MULTIPLY,
           '*': POWER, '⍟': LOGBASE, '|': RESIDUE, '⌈': CEILING,
           '⌊': FLOOR, '○': CIRCLE, '=': COMPEQ, '≠': COMPNEQ,
           '<': COMPLES, '>': COMPGRE, '≥': COMPGREQ, '≤': COMPLESQ,
           '^': GENERALAND, '∨': GENERALAND, '⍴': RESHAPE, '⌽': HORROT,
           '⊤': ENCODE, '⊥': DECODE, '⍳': FIND,
           '⊖': VERTROT, '⊢': RIGHT, '⊣': LEFT
           }.get(func)

    if len(func) > 1:
        # This is a user function
        return applyuserfunc(func, a, w)
    if fun is None:
        raise NotImplementedError('Function not yet supported: ' + func)
    # TODO implement all of the built in functions
    if DEBUG_MODE:
        print('applydi: ' + str(func) + ' ' + str(a) + ' ' + str(w))

    applied = []
    if func in scalarFuncs:
        arg = typeargs(a, w)
        if arg == -1:
            raise RuntimeError('Mixed lengths used! a = ' + str(a) + ' & w = ' + str(w))
        elif arg == 1:
            return fun(a, w)
        elif arg == 2:
            first = not isscalar(a)
            templist = a if first else w
            applied = [float(fun(
                np.array([scalar if first else float(a)]),
                np.array([float(w) if first else scalar])))
                       for scalar in list(templist.flat)]  # Applies the function to each member individually
            return np.array(applied).reshape(templist.shape)
        elif arg == 3:
            shape = a.shape
            a, w = a.ravel(), w.ravel()
            applied = [float(fun(np.array([float(a.flat[i])]), np.array([float(w.flat[i])])))
                       for i in range(a.shape[0])]  # a.shape should be equal to w.shape
            return np.array(applied).reshape(shape)

    elif func in mixedFuncs:
        return fun(a, w)


def applymo(func, w):
    fun = {'÷': INVERT, '*': EEXP, '⍟': NATLOG, '|': ABS, '○': PITIMES,
           '⍳': COUNT, '⍴': SHAPE, '~': BOOLNOT, '⍉': TRANSPOSE, '⊖': VERTFLIP,
           '⌽': HORFLIP, '⌈': ROUNDUP, '⌊': ROUNDDOWN, '?': RANDOM, '⊢⊣': ID}.get(func)

    if len(func) > 1:
        # User function
        return applyuserfunc(func, w=w)
    if fun is None:
        raise NotImplementedError('Function not yet supported: ' + func)

    if DEBUG_MODE:
        print('applymo: ' + str(func) + ' ' + str(w))

    if func in scalarFuncs + '~?':
        if w.shape == 0:
            return fun(w)
        else:
            return np.array([float(fun(np.array(scalar))) for scalar in list(w.flat)])

    elif func in mixedFuncs:
        return fun(w)  # Just send the entire thing to the function


def adverb(adv, func, w, userfunc=None):
    if DEBUG_MODE:
        print('adverb: ' + str(adv) + ' ' + str(func) + ' ' + str(w))
    # For lined arguments, transpose the array then at the end transpose it back
    ret = np.copy(w)  # Do not modify in place
    if adv in '\\/':
        ret = ret.transpose()

    if adv in '⌿/':
        for index, item in enumerate(ret):
            arr = item if isinstance(item, np.ndarray) else np.array([item])
            rtot = arr if index == 0 else applydi(func, rtot, arr)
        ret = rtot
    elif adv in '⍀\\':
        # This is where APL does the 'partial sums' of items
        for index, item in enumerate(ret):
            if index == 0:
                rtot = item
            else:
                rtot = applydi(func, rtot, item)
                ret[index] = rtot

    if adv in '/\\':
        ret = ret.transpose()

    if adv == '¨':
        # This is the APL 'each' operator
        # It applies a monadic function to each value of a vector
        # TODO: Test with dimensions
        if userfunc is None:
            return np.array([list(applymo(func, np.array([b])))[0] for b in ret])
        else:
            return np.array([list(applyuserfunc(userfunc, w=np.array([b])))[0] for b in ret])

    return ret


aplnamespace = {}


def apl(string, funcargs=[]):
    out = []
    for str in string.split(
            '\n'):  # if '\n' not in string, split will return a 1 element list containing string, so there is no need for separate case
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
        if DEBUG_MODE:
            print("\n")
            print("⍝⍝ NS ⍝⍝   " + str(aplnamespace))
            print("⍝⍝ OP ⍝⍝   " + str(opstack))
            print("⍝⍝ PD ⍝⍝   " + str(ParsingData))
            print("⍝⍝ TK ⍝⍝   " + str(token))

        hideOutp = False

        if token.type == 'COMMENT':
            continue

        if token.type == 'STATSEP':
            if len(opstack) == 1:  # We have a leftover op
                if opstack[0] != '←':
                    ParsingData = applymo(opstack.pop(), ParsingData)  # This should be unnecessary
                else:
                    raise SyntaxError('Syntax Error!')
            # Clear all of the data between statements
            stack = []

            opstack = []

            outofbracketdata = None

            hideOutp = False
            ParsingData = None
            continue

        if token.type == 'INDEX':
            results = []
            dimens = re.findall(r'[\[;][^;]+', token.value)
            # Trim the string to remove the semicolons / brackets
            results = [dim[1:-1] if ']' in dim else dim[1:] for dim in dimens]
            outofbracketdata = list(map(apl_wrapped, results))

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
                raise RuntimeError('Unmatched parens. Unable to continue')
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

                if token.type in ['PRIMFUNC', 'FUNLIT', 'ASSIGN']:
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

                if token.type in ['NUMBERLIT', 'VECTORLIT']:
                    if opstack[-1] == opassign:
                        raise RuntimeError('Attempting to assign a value to a constant, not a symbolic name.')
                    # This is the case when it is literal operation value
                    # e.g.: 5 * x
                    ParsingData = applydi(opstack.pop(), token.value, ParsingData)
                    continue

                elif token.type == 'FUNCARG':
                    if token.value == '⍺':
                        if len(funcargs) != 2:
                            raise ValueError('Attempted to use a diadic function monadically')
                        else:
                            ParsingData = applydi(opstack.pop(), funcargs[1], ParsingData)
                    elif token.value == '⍵':
                        if len(funcargs) == 0:
                            raise SyntaxError('Function arguments used outside of literal context!')
                        ParsingData = applydi(opstack.pop(), funcargs[0], ParsingData)

                elif token.type == 'NAME':
                    if opstack[-1] == opassign:
                        if len(opstack) == 2:
                            # This is the case when we are assigning a function to a value
                            aplnamespace[token.value] = opstack[0]
                            opstack = [opstack[1]]
                            continue
                        # Assign the parsing data to the value in the namespace
                        if DEBUG_MODE:
                            print("⍝⍝ AS ⍝⍝   " + str(token.value) + " := " + str(ParsingData))
                        aplnamespace[token.value] = ParsingData
                        opstack.pop()
                        hideOutp = True
                        continue
                    if not token.value in aplnamespace:
                        if DEBUG_MODE:
                            print('Referring to unassigned variable : ' + token.value + ' [will use 0]')
                        ParsingData = applydi(opstack.pop(), 0, ParsingData)
                    else:
                        get = aplnamespace[token.value]
                        if isinstance(get, np.ndarray):
                            ParsingData = applydi(opstack.pop(), get, ParsingData)
                        else:
                            # The name must be a function
                            # Apply the previous function as a monad ONLY if there's no adverb
                            if opstack[-1] in adverbs:
                                ParsingData = adverb(opstack.pop(), token.value, ParsingData, userfunc=get)
                                continue
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
                        raise SyntaxError("Syntax Error!")
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


introText = \
    "PyAPL version '0.2.2'  |  Created by Matt Torrence  |  Interactive Mode\n" \
    "                          Interactive Mode                               " \
    "--------------------  github.com/Torrencem/PyAPL  ---------------------\n"


def launchAPL():
    print(introText)
    while True:
        try:
            code = apl(input('APL>'))
        except KeyboardInterrupt:
            break
        # Comparison with 'in' does not work
        # with None
        if code is not [] and code is not None:
            print(code)


if __name__ == '__main__':
    launchAPL()
