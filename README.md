# PyAPL
A python implementation of the APL programming language

NOTE: This is not yet a complete project. So far, it works with fairly simple code samples. There is a LOT of features and testing that must be added

To install, use pip:

	pip install PyAPL

Example implementation:

    from PyAPL import *
    print(apl('(÷5-7)+÷15'))
    print(apl('((⍳4) × 4)<(7 + 4.2 1.4 1 5 × ⍳4)'))
    print(apl('(5 5⍴ ⍳4)>(5 5⍴ ⍳5)'))

Features yet to be added (in no particular order):

User-defined functions
goto's (→)
Axis-Bracket notation
A couple of built-in functions
Reading from .APL files
Boxing/full nesting (need to figure out how to implement correctly)