# code for running test cases, producing HTML, etc.

from .. import pprint_ex
from .. import build as build_uw
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class TestSuite(object):

    def __init__(self, build_in=None, fn='test_suite.agg', out_dir_name=''):
        """
        Run test suite fn. Create specified objects. Save graphics and info to HTML. Wrap
        HTML with template.

        TODO: convert wrapping to Jinja!

        To run whole test_suite::

            python -m aggregate.extensions.test_suite

        :param build_in: build object, allows input custom build object
        :param fn: test suite file name, default test_suite.agg
        :param out_dir_name: output directory name, default site_dir/generated
        """

        self.build = build_in if build_in else build_uw

        if out_dir_name != '':
            self.out_dir = Path(out_dir_name)
            if self.out_dir.exists() is False:
                raise FileExistsError(f'Directory {out_dir_name} does not exist.')
        else:
            self.out_dir = self.build.site_dir.parent / 'generated'
            self.out_dir.mkdir(exist_ok=True)
            (self.out_dir / "img").mkdir(exist_ok=True)

        logger.info(f'Output directory {self.out_dir.resolve()}')

        suite = self.build.default_dir / fn
        assert suite.exists(), f'Requested test suite file {suite} does not exist.'
        txt = suite.read_text(encoding='utf-8')
        tests = [i for i in txt.split('\n') if re.match(r'# [A-Z]\.', i)]
        self.tests = [i.replace("# ", "").split('. ') for i in tests]

    def run(self, regex, title, filename, browse=False, fig_format='svg', fig_size=(8,2.4), **kwargs):
        """
        Run all tests matching regex. Save graphics and info to HTML.
        Wrap HTML with template. To run whole test_suite use::

            python -m aggregate.extensions.test_suite

        :param regex: regex of tests to run, e.g., 'agg [ABC]\. '
        :param title: title for blob
        :param filename: file name prefix for saved immage files (convenience)
        :param browse: open browser to output file
        :param fig_format:  html or markdown (md); html uses svg output, markdown uses pdf
        :param fig_size:
        :param kwargs: passed to savefig
        """
        ans = []
        for n in self.build.qshow(regex, tacit=False).index:
            a = self.build(n)
            ans.append(a.html_info_blob().replace('h3>', 'h2>'))
            ans.append(pprint_ex(a.program, 50, True))
            ans.append(self.style_df(a.describe).to_html())
            ans.append('<br>')
            fn = self.out_dir / f'img/{filename}_tmp_{hash(a):0x}.{fig_format}'
            a.plot(figsize=fig_size)
            a.figure.savefig(fn, **kwargs)
            ans.append(f'<img src="{fn.resolve()}" />')
            plt.close(a.figure)
            logger.warning(f'Created {n}, mean {a.agg_m:.2f}')
        blob = '\n'.join([i if type(i)==str else i.data for i in ans])
        fn = self.out_dir / f'{filename}.html'
        fn.write_text(blob, encoding='utf-8')

        fn2 = self.out_dir / f'{fn.stem}_wrapped.html'
        fn3 = self.build.template_dir / 'test_suite_template.html'
        # TODO JINJA!
        template = fn3.read_text()
        template = template.replace('HEADING GOES HERE', title).replace(
            'CONTENTHERE', blob)
        fn2.write_text(template, encoding='utf-8')
        logger.info(f'Output written to {fn2.resolve()}')
        if browse:
            import webbrowser
            webbrowser.open(fn2.resolve().as_uri())

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


def run_test_suite():
    t = TestSuite()
    # show progress
    t.build.logger_level(30)
    print(t.tests)
    # run all the aggs
    # TODO FIX for Portfolios
    # t.run(regex=r'^C\.', title='C only', fig_prefix="auto", fig_format='png', dpi=300)
    t.run(regex=r'^[A-KNO]', title='Full Test Suite', filename='A_tests', browse=True,
          fig_format='png', dpi=300)


if __name__ == '__main__':
    run_test_suite()
