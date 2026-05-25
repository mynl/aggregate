# misc formatting

# bunch of lazy formatters to pass to styler
fp = lambda x: f'{x:.1%}'
fp3 = lambda x: f'{x:.3%}'
fc = lambda x: f'{x:.1g}'
fcm = lambda x: f'{x/1e6:,.1g}'
fg = lambda x: f'{x:8g%}'


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