# code for running test cases, producing HTML, etc.

from .. import pprint_ex

# from ..aggregate.utilities  import iman_conover, mu_sigma_from_mean_cv
# # from aggregate.utils import rearrangement_algorithm_max_VaR
# from .. aggregate.utilities import random_corr_matrix
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import re


logger = logging.getLogger(__name__)


class TestSuite(object):

    p = None
    build = None
    tests = ''

    @classmethod
    def __init__(cls, build=None, fn='test_suite.agg', dir_name='s/telos/blog/agg/generated'):
        """

        Run test suite fn. Create specified objects. Save graphics and info to HTML. Wrap
        HTML with template.

        TODO: convert wrapping to Jinja!

        """

        if build is None:
            from .. import build

        cls.build = build
        cls.p = Path.home() / dir_name

        # extract from comments
        suite = build.default_dir / fn
        txt = suite.read_text(encoding='utf-8')
        tests = [i for i in txt.split('\n') if re.match(r'# [A-Z]\.', i)]
        cls.tests = [i.replace("# ", "").split('. ') for i in tests]

    @classmethod
    def run(cls, regex, title, fig_format='svg', fig_size=(8,2.4)):
        """

        :param regex: regex of tests to run, e.g., 'agg [ABC]\. '
        :param title: title for blob
        :param fig_format:  html or markdown (md); html uses svg output, markdown uses pdf
        :param fig_size:
        """
        fig_prefix = re.sub(r'\^|\$|\[|\]|\*|\+', '', regex)
        logger.warning(f'figure prefix = {fig_prefix}')

        ans = []
        for n in cls.build.qshow(regex).index:
            a = cls.build(n)
            ans.append(a.html_info_blob().replace('h3>', 'h2>'))
            ans.append(pprint_ex(a.program, 50, True, True))
            ans.append(cls.style_df(a.describe).to_html())
            ans.append('<br>')
            fn = cls.p / f'img/{fig_prefix}_tmp_{hash(a):0x}.{fig_format}'
            a.plot(figsize=fig_size)
            a.figure.savefig(fn)
            ans.append(f'<img src="{fn.resolve()}" />')
            plt.close(a.figure)
            logger.warning(f'Created {n}, mean {a.agg_m:.2f}')

        blob = '\n'.join(ans)
        fn = cls.p / f'{fig_prefix}.html'
        fn.write_text(blob, encoding='utf-8')

        fn2 = cls.p / f'{fn.stem}_wrapped.html'
        fn3 = cls.p / 'template.html'
        template = fn3.read_text()
        template = template.replace('HEADING GOES HERE', title).replace(
            'CONTENTHERE', blob)
        fn2.write_text(template, encoding='utf-8')

    @staticmethod
    def style_df(df):
        """
        Style a df similar to pricinginsurancerisk.com styles.

        graph background color is B4C3DC and figure (paler) background is F1F8F#

        Dropped row lines; bold level0, caption

        :param df:
        :return: styled dataframe

        """

        cell_hover = {
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        index_names = {
            'selector': '.index_name',
            'props': 'font-style: italic; color: black; background-color: white; '
                     'font-weight:bold; border: 0px solid #a4b3dc; text-transform: capitalize; '
                     'text-align:left;'
        }
        headers = {
            'selector': 'th:not(.index_name)',
            'props': 'background-color: #DDDDDD; color: black;  border: 1px solid #ffffff;'
        }
        center_heading = {
            'selector': 'th.col_heading',
            'props': 'text-align: center;'
        }
        left_index = {
            'selector': '.row_heading',
            'props': 'text-align: left;'
        }
        td = {
            'selector': 'td',
            'props': f'text-align: right; '
        }
        nrow = {
            'selector': 'tr:nth-child(even)',
            'props': 'background-color: #F5F5F5;'
        }
        all_styles = [cell_hover, index_names, headers, center_heading, nrow, left_index, td]

        fc = lambda x: f'{x:,.3f}' if isinstance(x, (float, int)) else x
        f3 = lambda x: f'{x:.3f}' if isinstance(x, (float, int)) else x
        f5g = lambda x: f'{x:.5g}' if isinstance(x, (float, int)) else x
        # guess sensible defaults
        fmts = {'E[X]': fc,
                'Est E[X]': fc,
                'Err E[X]': f5g,
                'CV(X)': f3,
                'Est CV(X)': f3,
                'Err CV(X)': f5g,
                'Skew(X)': f3,
                'Est Skew(X)': f3}
        return df.style.set_table_styles(all_styles).format(fmts)
