from pathlib import Path
import re


# bunch of lazy formatters to pass to styler
fp = lambda x: f'{x:.1%}'
fp3 = lambda x: f'{x:.3%}'
fc = lambda x: f'{x:.1g}'
fcm = lambda x: f'{x/1e6:,.1g}'
fg = lambda x: f'{x:8g%}'


def rst_to_md(fn):
    """
    Strip ipython code out of rst and make an md file that can be read into Jupyter
    """
    if type(fn) == str:
        fn = Path(fn)
    assert fn.exists()

    txt = fn.read_text(encoding='utf-8')
    # can do this in one line but it is incomprehensible
    # strictly four space tabs: split on ipython: , pull out right parts; remove leading tabs, remove @savefig
    stxt = re.split(r'.. ipython:: +python\n( +:okwarning:\n)?( +:suppress:\n)?', txt)[3::3]
    code0 = [re.split('\n\n+', s)[0] for s in stxt]
    code1 = [re.sub('^( |\t)*@savefig[^\n]+\n', '', s, flags=re.MULTILINE) for s in code0]
    python_code = [i.strip().replace('\n    ', '\n') for i in code1]

    # reassemble
    preamble = '```python\n'
    postamble = '\n```'
    str_out = preamble + '\n```\n\n```python\n'.join(python_code) + postamble
    fout = (Path('/s/telos/temp/z') / fn.name).with_suffix('.md')
    fout.write_text(str_out, encoding='utf-8')
    return fout

