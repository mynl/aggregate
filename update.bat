@echo off

if %1% == u (
    echo uploading
    twine upload -u mildenhall -p %PPPW% dist/*0.9.6* --verbose
) else (
    echo building
    python setup.py bdist_egg
    python setup.py sdist
)
