REM mildenhall/ xx for PyPi see keepass

REM twine upload dist/aggregate-0.9.5.1.*

REM internet says
REM twine upload dist/*

echo %1

twine check dist/aggregate-%1*

echo twine upload --verbose -u mildenhall -p %PPPW% dist/*%1*

twine upload --verbose -u mildenhall -p %PPPW% dist/*%1*


