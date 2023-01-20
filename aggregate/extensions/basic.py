from pathlib import Path
import re


# bunch of lazy formatters to pass to styler
fp = lambda x: f'{x:.1%}'
fp3 = lambda x: f'{x:.3%}'
fc = lambda x: f'{x:.1g}'
fcm = lambda x: f'{x/1e6:,.1g}'
fg = lambda x: f'{x:8g%}'


def rst_to_md(fn, to_dir='/s/telos/temp/z'):
    """
    Strip ipython code out of rst file and make an md file that can be read into Jupyter
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
    fout = (Path(to_dir) / fn.name).with_suffix('.md')
    fout.write_text(str_out, encoding='utf-8')
    return fout


def all_rst_to_md(from_dir='doc', to_dir='/s/telos/temp/z'):
    """
    Convert all rst files in from_dir  to md files
    """

    if from_dir == 'doc':
        from_dir = Path(__file__).parent.parent.parent / 'doc'
        assert from_dir.exists()

    for fn in Path(from_dir).glob('**/*.rst'):
        print(f'Converting {fn}')
        rst_to_md(fn, to_dir)


class Formatter(object):
    def __init__(self, w=8, dp=3, pdp=1, threshold=1000):
        """
        dp for < threshold
        pdp for percentages, used if >=0
        """
        self.threshold = threshold
        if pdp >= 0:
            self.pdp = f'{{x:{w}.{pdp}%}}'
        else:
            self.pdp = None
        self.dp = f'{{x:{w},.{dp}f}}'
        self.big = '{x:{w},.0f}'

    def __call__(self, x):
        if type(x) == str:
            return x
        if self.pdp is not None and x <= 1:
            return self.pdp.format(x=x)
        elif abs(x) <= self.threshold:
            return self.dp.format(x=x)
        else:
            return self.big.format(x=x)