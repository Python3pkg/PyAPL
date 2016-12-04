import begin
import logging
logging.basicConfig(filename='PyPL.log',level=logging.DEBUG)
from src import APLex

# @begin.start
# def run(file: 'The input file path', output: 'The output file path'):
#
#     pass

def applydi(func, a, w):
    # TODO: For relevant functions, implement size checking functions for vectors
    # Also, implement correct vector [list] algorithms
    # and implement all of the built in functions
    logging.info('applydi: ' + str(func) + ' ' + str(a) + ' ' + str(w))
    if func == '+':
        return a + w
    if func == '-':
        return a - w

def applymo(func, w):
    logging.info('applymo: ' + str(func) + ' ' + str(w))
    if func == '/':
        return 1/w


def apl(string, useLPN = False): # useLPN = use Local Python Namespace (share APL functions and variables)
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
        logging.info('tk : ' + str(token.type) + '   ' + str(token.value))

        if token.type == 'RPAREN':
            stack.append((ParsingData, opstack))  # store both this parsing data and the opstack
            ParsingData = None
            opstack = []
            continue

        if token.type == 'LPAREN':

            if len(opstack) == 1:  # e.g.: (/3+4) - 2
                # Apply the last op as a monad
                ParsingData = applymo(opstack.pop(),ParsingData)

            if stack == []:
                logging.fatal('Unmatched parens. Unable to continue')
                raise RuntimeError()
            else:
                lStack = stack.pop()
                lopstack = lStack[1]
                if len(lopstack) == 1:  # e.g. : (3 + 5)/2
                    ParsingData = applydi(lopstack.pop(),ParsingData,lStack[0])
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
                    ParsingData = namespace[token.value] #TODO: Check if name is a function

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
                        ParsingData = applydi(opstack.pop(), namespace[token.value], ParsingData)  # TODO: Check if name is a function
                elif token.type == 'PRIMFUNC':
                    # Apply the first function as a monadic function then continue parsing
                    ParsingData = applymo(opstack.pop(), ParsingData)
                    opstack.append(token.value)

                #TODO: add extra token conditions here

    return ParsingData




if __name__ == '__main__':
    print(apl('(/5-7)+/15'))