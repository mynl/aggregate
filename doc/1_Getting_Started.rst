===================
Getting Started
===================

Installation
------------

Install from PyPI ::

   pip install aggregate

See https://pypi.org/project/aggregate/.


Source Code
-----------

The source code is hosted on GitHub, https://github.com/mynl/aggregate.


Documentation
-------------

The rest of the documentation is divided into five parts.

1. The :doc:`2_User_Guide` contains a basic introduction and shows how to use aggregate to solve different actuarial modeling problems.

2. :doc:`3_Reference` gives the class and function reference.

3. :doc:`4_agg_Language_Reference` describes the agg language, a human readable way to describe aggregate loss distributions aimed at insurance applications.

4. :doc:`5_Technical_Resources` provides actuarial, probability, and other non-programming background.

5. :doc:`6_Development` about history, ideas for the future.

The documentation borrows ideas from pandas.

Prerequisites
-------------

The help assumes you know how to program in Python, understand probability, and are familiar with the concept of an aggregate distribution. Awareness of the material covered in SOA exam STAM, CAS exam MAS I, or IFOA CS-2 is helpful. It also assumes you understand insurance terminology like limit, attachment, deductible.

Dependencies
------------

The Python library depends on numpy, pandas, matplotlib, scipy and uses Python 3.8 or higher.

The parser uses sly, a fantastic lex/yacc for Python, https://github.com/dabeaz/sly.


License
-------

BSD 3

