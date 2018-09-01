"""
manage built in and user portfolios, severity lines and access to parameters

"""

import matplotlib.pyplot as plt
from IPython.core.display import display
from copy import deepcopy
import logging
from ruamel import yaml
import os
from . utils import html_title, sensible_jump
from . port import Portfolio

class Example(object):
    """
    Example: class
    Manages examples for Portfolio object
    Building examples from built in lines
    """

    def __init__(self, dirname=""):
        """
        load filename or YAMLFILE if not given
        :param dirname:
        """
        if dirname == '':
            # TODO sort out...
            dirname = 'c:/s/telos/python/aggregate'
        with open(os.path.join(dirname, 'portfolios.yaml'), 'r') as f:
            self.built_in_portfolios = yaml.load(f, Loader=yaml.Loader)
        with open(os.path.join(dirname, 'lines.yaml'), 'r') as f:
            self.built_in_lines = yaml.load(f, Loader=yaml.Loader)
        with open(os.path.join(dirname, 'severity.yaml'), 'r') as f:
            self.built_in_severity = yaml.load(f, Loader=yaml.Loader)
        with open(os.path.join(dirname, 'user.yaml'), 'r') as f:
            self.user_portfolios = yaml.load(f, Loader=yaml.Loader)
        self.spec_list = []

    @staticmethod
    def rescale_sev(spec, scale):
        """
        Apply scale to a built in line
        :param scale:
        :param spec:
        :return:
        """
        if scale > 0:
            if 'mean' in spec['severity']:
                spec['severity']['mean'] *= scale
            elif 'scale' in spec['severity']:
                spec['severity']['scale'] *= scale
            else:
                logging.error(f'example | Cannot adjust {spec["name"]} for scale shift {scale}')
            if 'loc' in spec['severity']:
                # for Pareto
                spec['severity']['loc'] *= scale
            if 'limit' in spec:
                spec['limit'] *= scale

    @staticmethod
    def shift_sev(spec, shift):
        """
        shift severity for a built in line, leaves limit unchanged
        :param shift:
        :param spec:
        :return:
        """
        if shift != 0:
            if 'loc' in spec['severity']:
                spec['severity']['loc'] += shift
            else:
                logging.error(f'example | Cannot adjust {spec["name"]} for scale shift {shift}')

    @staticmethod
    def adjust_frequency(spec, scale=0, shift=0):
        """
        Adjust scale and shift freq for a built in line
        can make both adjustment, scale first then shift
        :param spec:
        :param scale:
        :param shift:
        :return:
        """
        if scale > 0:
            spec['frequency']['n'] *= scale
        if shift != 0:
            spec['frequency']['n'] += shift

    def list(self):
        """
        list all available portfolios and lines
        :return:
        """
        print('\n\t- '.join(['Built in portfolios'] + list(self.built_in_portfolios.keys())))
        print()
        print('\n\t- '.join(['Built in lines'] + list(self.built_in_lines.keys())))
        print()
        print('\n\t- '.join(['Built in severity'] + list(self.built_in_severity.keys())))
        print()
        print('\n\t- '.join(['User portfolios'] + list(self.user_portfolios.keys())))

    def portfolio(self, portfolio, reporting_level=0, **kwargs):
        """
        make an example by name, convenience function
        includes Bodoff, Kent and sensible default examples
        kwargs passed through to CPortfolio.update

        :param portfolio: pre-defined portfolio name
        :param reporting_level: what post-construction reports to print (mostly graphcs), 0 to 3, higher is more detail
        :param kwargs:    additional args passed through to update; note: if  line_list this must
                include log2 and bs
        :return:  portfolio object, recomputed from spec
        """

        try:
            if portfolio in self.user_portfolios:
                params = self.user_portfolios[portfolio]
            else:
                params = self.built_in_portfolios[portfolio]
        except KeyError:
            logging.error(f'Example.portfolio | unknown portfolio {portfolio} requested ')
            raise KeyError

        port = Portfolio(portfolio, params['spec'])
        # update if there is enough information
        if 'log2' in params['args'] and 'bs' in params['args']:
            port.update(**{**params['args'], **kwargs})
            log2 = params['args']['log2']
            Example.reporting(port, log2, reporting_level)
        return port

    def new(self):
        """
        create a new example; just clears out the old spec
        then call addline...
        :param self:
        :return:
        """
        self.spec_list = []

    def addline(self, line_name, freq_scale=1, freq_shift=0, sev_scale=1, sev_shift=0):
        """
        Sample using built in lines (minimal)
        ex.addline('home', 1000)
        ex.addline('auto', 3000)
        ex.addline('olocc', 500)
        ex.publish(13, 10000)

        :param line_name:
        :param freq_scale:
        :param freq_shift:
        :param sev_scale:
        :param sev_shift:
        :return:
        """

        try:
            spec = deepcopy(self.built_in_lines[line_name])
        except KeyError:
            logging.error(f'Example.addline | attempted to add unknown line {line_name}')
            raise KeyError(f'{line_name} is not a known built in line')
        Example.adjust_frequency(spec, freq_scale, freq_shift)
        Example.rescale_sev(spec, sev_scale)
        Example.shift_sev(spec, sev_shift)
        self.spec_list.append(spec)
        logging.info(f'Example.addline | added line {line_name}')

    def publish(self, name, log2=0, bs=0, reporting_level=0, **kwargs):
        """
        complete construction of example made from built in lines

        :param name:
        :param log2:
        :param bs:
        :param reporting_level:
        :param kwargs:
        :return:
        """
        port = Portfolio(name, self.spec_list)
        if log2 == 0 and bs == 0:
            # used to estimate moments, no update
            html_title('Theoretical Statistics')
            display(port.statistics_df)
            html_title('Bucket Recommendations')
            display(port.recommend_bucket().style)
        else:
            port.update(log2=log2, bs=bs, **kwargs)
            Example.reporting(port, log2, reporting_level)
        logging.info(f'Example.publish | publishing object {port.name} with {len(port.agg_list)} lines')
        return port

    def line(self, name):
        """
        return dictionary line specification, for building portfolios
        :param name:
        :return:
        """
        if name in self.built_in_lines:
            logging.info(f'Example.line | serving {name}')
            return self.built_in_lines[name]
        else:
            logging.warning(f'Example.line | failed to find {name}')
            print(f'{name} is not available as a line')

    def severity(self, name):
        """
        return dictionary severity specification, for building portfolios

        :param name:
        :return:
        """
        if name in self.built_in_severity:
            logging.info(f'Example.severity | serving {name}')
            return self.built_in_severity[name]
        else:
            logging.warning(f'Example.severity | failed to find {name}')
            print(f'{name} is not available as a severity')

    def __getitem__(self, name):
        """
        make Example scriptable: try user portfolios, b/in portfolios, line, severity
        to access specifically use severity or line methods
        :param name:
        :return:
        """
        if name in self.user_portfolios:
            logging.info(f'Example.__getitem__ | serving user portfolio {name}')
            return self.portfolio(name, 1)
        elif name in self.built_in_portfolios:
            logging.info(f'Example.__getitem__ | serving built in portfolio {name}')
            return self.portfolio(name, 1)
        elif name in self.built_in_lines:
            logging.info(f'Example.__getitem__ | serving line {name}')
            return self.built_in_lines[name]
        elif name in self.built_in_severity:
            logging.info(f'Example.__getitem__ | serving severity {name}')
            return self.built_in_severity[name]
        else:
            logging.error('Example.__getitem__ | unknown object {name} requested')
            raise KeyError

    @staticmethod
    def reporting(port, log2, reporting_level):
        """
        handle various reporting options: most important to appear last

        :param port:
        :param log2:
        :param reporting_level:
        :return:
        """

        if reporting_level >= 3:
            # just plot the densities
            f, axs = plt.subplots(1, 6, figsize=(15, 2.5))
            axiter = iter(axs.flatten())
            port.plot(kind='quick', axiter=axiter)
            port.plot(kind='density', line=port.line_names_ex, axiter=axiter)
            port.plot(kind='density', line=port.line_names_ex, axiter=axiter, legend=False, logy=True)
            plt.tight_layout()

        if reporting_level >= 2:
            jump = sensible_jump(2 ** log2, 10)
            html_title('Line Densities', 1)
            display(port.density_df.filter(regex='^p_[^n]|S|^exa_[^n]|^lev_[^n]').
                    query('p_total > 0').iloc[::jump, :])
            html_title('Audit Data', 1)
            display(port.audit_df.filter(regex='^[^n]', axis=0))

        if reporting_level >= 1:
            html_title('Summary Audit Data', 1)
            temp = port.audit_df.filter(regex='^Mean|^EmpMean|^CV|^EmpCV')
            temp['MeanErr'] = temp.EmpMean / temp.Mean - 1
            temp['CVErr'] = temp.EmpCV / temp.CV - 1
            temp = temp[['Mean', 'EmpMean', 'MeanErr', 'CV', 'EmpCV', 'CVErr']]
            display(temp.style.applymap(Example.highlight))

        if reporting_level >= 3:
            html_title('Graphics', 1)

    @staticmethod
    def highlight(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for large
        values. As pct error
        """
        color = 'red' if abs(val) > 0.01 and val < 1 else 'black'
        return f'color: {color}; font-weight: bold'
