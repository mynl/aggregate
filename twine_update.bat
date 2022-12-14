REM mildenhall/ xx for PyPi see keepass

REM twine upload dist/aggregate-0.9.5.1.*

REM internet says
REM twine upload dist/*

twine check dist/aggregate-0.9.6.1*

twine upload -u mildenhall -p %PPPW% dist/*0.9.6.1* --verbose


