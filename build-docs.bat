REM doc build checker 

REM clean old attempts 
rmdir /s /q c:\tmp\agg-test

REM target 
mkdir c:\tmp\agg-test
cd \tmp\agg-test

REM clone local rep 
git clone --no-single-branch --depth 50 file:///c:/S/TELOS/python/aggregate_project .

REM don't want this: allow uncommitted changes to roll over 
REM git checkout --force origin/master

REM clean up: removes untracked files and directories. 
REM git clean -d -f -f
REM see what it would do, -n = dry run 
REM git clean -d -n


REM create venv 
python -mvirtualenv ./venv

REM activate the virtual environment
venv\Scripts\activate.bat

REM install agg in dev mode 
pip install aggregate[dev]

REM try making the docs - just HTML is enough
cd docs 
call make html 


 
