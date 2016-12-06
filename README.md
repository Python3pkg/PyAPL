# PyAPL
A python implementation of the APL programming language

NOTE: This is in no way a complete project. So far, it works with fairly simple code samples. There is a LOT of features and testing that must be added

A file mode is yet to be implemented.

Example implementation:

    from PyAPL import *
    print(apl('(÷5-7)+÷15'))
    print(apl('(÷1 253 3) - (÷3 2 1)'))
    print(apl('(1 2 3 4 × 4)<(7 + 4 2 1 5 × ⍳4)'))
    print(apl('((⍳4) × 4)<(7 + 4.2 1.4 1 5 × ⍳4)'))