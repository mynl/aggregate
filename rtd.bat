

REM clear out current env variable for this session only
set PYTHONPATH=

REM git clone --no-single-branch --depth 50 https://github.com/mynl/aggregate.git github

rem cd github

cd \temp\rtd

rmdir /s local /q

rem clone local copy
git clone --no-single-branch --local --depth 50 file://c:\s\telos\python\aggregate_project local

cd local

git clean -d -f -f


REM env to run and install it
call conda create --name agg-build python=3.8 -y

call conda activate agg-build

python -m pip install --upgrade --no-cache-dir pip "setuptools<58.3.0"

python -m pip install --upgrade --no-cache-dir pillow mock==1.0.1 "alabaster>=0.7,<0.8,!=0.7.5" commonmark==0.9.1 recommonmark==0.5.0 "sphinx<2" "sphinx-rtd-theme<0.5" readthedocs-sphinx-ext "jinja2<3.1.0"

python -m pip install --exists-action=w --no-cache-dir -r requirements.txt

REM cat doc/conf.py
REM not sure what this was all about??!

cd doc


python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html > sphinx-build.md 2>&1


"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" C:\temp\rtd\local\doc\_build\html\index.html

REM clean up
REM conda activate base
REM conda env remove -n agg-build -y
