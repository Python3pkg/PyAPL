# PyAPL
A python implementation of the APL programming language

For more info on the language, check out the dyalog reference: http://docs.dyalog.com/13.0/Dyalog%20APL%20Language%20Reference.v13.0.pdf

or, look at the legendary APL dictionary: http://www.jsoftware.com/papers/APLDictionary.htm

Also, you can run APL in your browser! http://tryapl.org/ [highly recommended if you don't have a keyboard setup!]

NOTE: This is not yet a complete project. So far, it works with fairly simple code samples.

To install, use pip:

	pip install PyAPL

Example implementation:

    from PyAPL import *
    print(apl('(÷5-7)+÷15'))
    print(apl('((⍳4) × 4)<(7 + 4.2 1.4 1 5 × ⍳4)'))
    print(apl('(5 5⍴ ⍳4)>(5 5⍴ ⍳5)'))

To launch an interactive APL prompt, do the following:

    pip install PyAPL
    ...
    python

    >>>from PyAPL import *
    >>>launchAPL()

Features yet to be added (in no particular order):

goto's (→)

Axis-Bracket notation

A couple of built-in functions

Reading from .APL files

Boxing/full nesting (need to figure out how to implement correctly)

More syntax error catchers. This implementation might not be the best for someone very new to APL, since it won't catch some mistakes easily yet.


NOTE: One gotcha of this interpreter is that it reads multi-statements right to left, like they probably should be. An example is found in the function unittest.
