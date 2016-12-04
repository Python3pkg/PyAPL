import begin
import logging

logging.basicConfig(filename='PyPL.log', level=logging.DEBUG)
from src import APLex
from collections import namedtuple

# NOTE: If an APLobj's size is 0,
# it's value should be an INT, not a list
APLobj = namedtuple('Data', 'value, shape')

# Scalar functions apply their function to each of the parts of a vector
# individually
scalarFuncs = '+ - × ÷ | ⌈ ⌊ * ⍟ ○ ! ^ ∨ ⍲ ⍱ < ≤ = ≥ > ≠'

# Mixed functions do something else
# NOTE that the monadic versions of ~ and ? are scalar, while the diadic versions
# are actually mixed.
mixedFuncs = '⍴ , ⍪ ⌽ ⊖ ⍉ ↑ ↓ / ⌿ \ ⍀ ⍳ ∊ ⍋ ⍒ ? ⌹ ⊥ ⊤ ⍕ ⍎ ⊂ ⊃ ≡ ⍷ ⌷ ~ ?'

# Returns: False for unmatched types [2 3 4 = 3 4]
# 1 for scalar scalar [3 + 43]
# 2 for scalar table/vector [3 - 14 23 11] OR [41 12 1 > 4]
# 3 for the same shape table/vector [3 2 3 > 1 4 1]
def typeargs(a, w):
    if a.shape != 0 and w.shape != 0:
        if a.shape != w.shape:
            return -1
        else:
            return 3
    elif a.shape != 0 or w.shape != 0:
        return 2
    else:
        return 1


def subapplydi(func, a, w):
    if func == '+':
        return APLobj(a.value + w.value,0)
    elif func == '-':
        return APLobj(a.value - w.value,0)
    elif func == '÷':
        return APLobj(a.value / w.value,0)
    elif func == '×':
        return APLobj(a.value * w.value,0)
    elif func == '=':
        return APLobj((1 if a.value == w.value else 0),0)
    elif func == '<':
        return APLobj((1 if a.value < w.value else 0),0)
    elif func == '>':
        return APLobj((1 if a.value > w.value else 0),0)
    elif func == '≥':
        return APLobj((1 if a.value >= w.value else 0),0)
    elif func == '≤':
        return APLobj((1 if a.value <= w.value else 0),0)
    else:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()


def applydi(func, a, w):
    # TODO implement all of the built in functions
    logging.info(('applydi: ' + str(func) + ' ' + str(a) + ' ' + str(w)).encode('utf-8'))
    applied = APLobj([],0)
    if func in scalarFuncs:
        arg = typeargs(a, w)
        if arg == -1:
            logging.fatal('Mixed lengths used! a = ' + str(a) + ' & w = ' + str(w))
            raise RuntimeError()  # TODO: pretty up error messages
        elif arg == 1:
            return subapplydi(func, APLobj(a.value,0), APLobj(w.value,0))
        elif arg == 2:
            first = True if a.shape != 0 else False
            templist = a if first else w
            tempscal = a if not first else w
            for scalar in templist.value:  # Applies the function to each member individually
                applied.value.append(subapplydi(func,
                                                APLobj(scalar,0) if first else APLobj(tempscal.value,0),
                                                APLobj(scalar,0) if not first else APLobj(tempscal.value,0)).value)
            applied = APLobj(applied.value, templist.shape)
        elif arg == 3:
            for i in range(0, len(a.value)):  # len(a.value) should be equal to len(w.value)
                applied.value.append(subapplydi(func, APLobj(a.value[i],0), APLobj(w.value[i],0)).value)
            applied = APLobj(applied.value, a.shape)
        return applied

    elif func in mixedFuncs:
        pass


def subapplymo(func, w):
    if func == '÷':
        return APLobj(1 / w.value,0)
    elif func == '⍳':
        # Ioda
        ### "COUNT" function ###
        intermed = []
        if w.shape != 0:
            logging.fatal("A vector has been passed to ⍳ function. Undefined behavior!")
            return w  # Just to do something
        else:
            for i in range(w.value):
                intermed.append(i)
        return APLobj(intermed, len(intermed))
    elif func == '⍴':
        # Rho
        ### "SIZE" function ###
        if isinstance(w.shape, int):
            return APLobj(w.shape, 0)
        else:
            # When it is multi-dimensional
            return APLobj(w.shape, len(w.shape))
    else:
        logging.error('Function not yet supported: ' + func)
        raise NotImplementedError()  # TODO: continue implementing primitive functions


def applymo(func, w):
    logging.info(('applymo: ' + str(func) + ' ' + str(w)).encode('utf-8'))
    applied = APLobj([],0)
    if func in scalarFuncs:
        if w.shape == 0:
            return subapplymo(func, w)
        else:
            for scalar in w.value:
                applied.value.append(subapplymo(func, APLobj(scalar,0)).value)
            applied = APLobj(applied.value, w.shape)
            return applied

    elif func in mixedFuncs:
        return subapplymo(func, w)  # Just kind of do it



def apl(string, useLPN=False):  # useLPN = use Local Python Namespace (share APL functions and variables) TODO [NYI]
    lex = APLex.APLexer()
    lex.build()
    logging.info('Parsing string... len = ' + str(len(string)))
    tokens = lex.inp(string)
    # APL is a right to left language, so we will reverse the token order
    tokens = tokens[::-1]
    logging.info('Parsing tokens... len = ' + str(len(tokens)))
    ParsingData = None

    namespace = {}

    stack = []

    opstack = []

    for token in tokens:
        logging.info(('tk : ' + str(token.type) + '   ' + str(token.value)).encode('utf-8'))

        if token.type == 'RPAREN':
            stack.append((ParsingData, opstack))  # store both this parsing data and the opstack
            ParsingData = None
            opstack = []
            continue

        if token.type == 'LPAREN':

            if len(opstack) == 1:  # e.g.: (/3+4) - 2
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

                if not token.value in namespace:
                    ParsingData = 0
                    logging.error('Referring to unassigned variable : ' + token.value + ' [will assign 0]')
                    namespace[token.value] = 0
                else:
                    ParsingData = namespace[token.value]  # TODO: Check if name is a function

            elif token.type == 'NUMBERLIT' or token.type == 'VECTORLIT':
                ParsingData = token.value
        else:

            if len(opstack) == 0:

                if token.type == 'PRIMFUNC':
                    opstack.append(token.value)
                    continue
                elif token.type == 'ASSIGN':
                    opstack.append(token.value)
                    continue

            elif len(opstack) == 1:

                if token.type == 'NUMBERLIT' or token.type == 'VECTORLIT':
                    # This is the case when it is literal operation value
                    # e.g.: 5 * x
                    ParsingData = applydi(opstack.pop(), token.value, ParsingData)
                elif token.type == 'NAME':
                    if not token.value in namespace:
                        logging.error('Referring to unassigned variable : ' + token.value + ' [will use 0]')
                        ParsingData = applydi(opstack.pop(), 0, ParsingData)
                    else:
                        ParsingData = applydi(opstack.pop(), namespace[token.value],
                                              ParsingData)  # TODO: Check if name is a function
                elif token.type == 'PRIMFUNC':
                    # Apply the first function as a monadic function then continue parsing
                    ParsingData = applymo(opstack.pop(), ParsingData)
                    opstack.append(token.value)
                    continue

    if len(opstack) == 1:  # We have a leftover op
        ParsingData = applymo(opstack.pop(), ParsingData)


                    # TODO: add extra token conditions here

    return ParsingData


if __name__ == '__main__':
    # print(apl('(÷5-7)+÷15'))
    # print(apl('2 3 4 + 1 2 1'))
    # print(apl('1 2 3 4 × 4'))
    # print(apl('7 + 4 2 1 5 × ⍳4'))
    # print(apl('(1 2 3 4 × 4)<(7 + 4 2 1 5 × ⍳4)'))
    # try:
    #     print(apl('4 2 1 5 × 1 2 3'))
    # except RuntimeError:
    #     print('test passed; mixed lengths error')
    # print(apl('(÷1 253 3) - (÷3 2 1)'))
    while(True):
        e = apl(input('>>>')).value
        if isinstance(e, list):
            print(' '.join(str(x) for x in e))
        else:
            print(e)