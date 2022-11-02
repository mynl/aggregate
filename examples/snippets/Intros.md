---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# V01: A First Introduction to `aggregate`

**Objectives** Introduce aggregate probability distributions and the `aggregate` library for working with them. 

**Audience** New user with no knowlege of aggregate distributions or insurance.

**Prerequisites** Basic probability theory; Python and pandas programming.

**Context** Up next: `aggregate` for Actuarial Students.

**Overview**
1. Definition of aggregate (compound) probability distributions.
2. Applications and examples.
3. Installing the `aggregate` package.
4. Simple discrete examples illustrating using dice throws.
5. Determining moments of and plotting aggregate distributions.


# V02: `aggregate` for Actuarial Students 

**Objectives** Introduce aggregate probability distributions and the `aggregate` library for working with them in the context of exam and university courses in actuarial modeling. 

**Audience** Actuarial science university student or junior analyst working in insurance.

**Prerequisites** V01 plus familiarity with aggregate probability distribution (as covered on SOA STAM, CAS MAS I, IFOA CS-2) and basic insurance terminology (insurance company operations).

**Context** Up next: `aggregate` for Actuaries.

**Overview**
1. Installing the `aggregate` package.
2. Working with simple discrete aggregate probability distributions.
3. Determining moments of and plotting aggregate distributions.


# V03: `aggregate` for Actuaries 

**Objectives** Introduce aggregate probability distributions and the `aggregate` library for working with then in the context of real-world, but basic, actuarial problems 
illustrated using the Tweedie distribution from GLM modeling. 

**Audience** Actuaries at the Associate or Fellow level.

**Prerequisites** V02 plus awareness of the use aggregate probability distribution in insurance (as covered on CAS Part 8).

**Context** Up next: `aggregate` for individual risk pricing actuaries.

**Overview**
1. Installing the `aggregate` package.
2. Computing an aggregate distribution for a portfolio of risks with given frequency, severity, limit, and attachment assumptions.
3. Different ways to determine exposure (prem x lr; claim count; losses).
4. Determining limited expected values.

<!-- #region -->
# Introductions to `aggregate`

| Audience                        | `Underwriter`   | Features and Prerequisites                                                                          | Problems                               |
|:--------------------------------|:----------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------|
| Novice                          | `student_build`  | Intro to aggregate distributions in general language; simple discrete examples                      |                                        |
| Actuarial students              | `student_bulid` | Similar to novice but using insurance terminology                                                   | SOA STAM, CAS MAS1, IFOA CS2, KPW, LDA |
| Actuaries                       | `actuary_build` | Introduction using the Tweedie distribution as motivation                                           | CAS Part 8                             |
| Individual risk pricing actuary | `actuary_build`      | LEV, ILFs, layering, aggregate insurance charge (Table L, M); solving problems from CAS             | CAS Part 8                             |
| Reinsurance pricing actuary     | `actuary_build`      | Exposure rating, swings and slides, aggregate stop loss                                             | CAS Part 8                             |
| Reserving actuary               | `actuary_build` | Loss emergence, IBNR and case reserve ranges                                                        |                                        |
| Capital modeler                 | `capital_build` | Use of samples, VaR, TVaR, tail evaluation, risk drivers; Iman-Conover; rearrangement algorithm     | CAS Part 9                             |
| Strategic planning              | `capital_build`   | Capital allocation in a portfolio; alternative pricing methodologies; bounds on net/gross pricing   | CAS Part 9                             |
| Catastrophe modeler             | `capital_build`     | Thick tailed Poisson Pareto and lognormal models; aggregate PMLs; occurrence and aggregate layering | CAS Part 9                             |
| Act Sci or Risk Mgmt professor  | `student_build` | Solving text book problems (similar to Student intro); generate realistic, motivating examples      |                                        |
| Developer                       | `dev_build`     | Class model, `agg` language grammar, internal design                                                |                                        |
| Debugger                        | `debug_build`   | Watch inner workings!                                                                               |                                        |                                                                                                |                                        |

For each audience there is a customized `Underwriter` object 

## Basic prequisies for all introductions

* Basic probability including discrete and continuous distributions, mean, variance, standard deviation, pdf, cdf. 
* Python programming, including familiarity with `pandas`. `numpy`, `matplotlib`, and `scipy.stats` useful but not essential.



## Provenance

* Oct 2022 created new 

<!-- #endregion -->

```python
p = Path.home() / 'aggregate/databases'
p = Path.home() / 'agg'
p = Path('/s/telos/python/aggregate_project/aggregate/agg')

```

```python
p.exists()
```

```python
entries = []
for fn in p.glob('*.agg'):
    txt = fn.read_text(encoding='utf-8')
    stxt = txt.split('\n')
    for r in stxt:
        rs = r.split(' ')
        if rs[0] in ['agg', 'port', 'dist', 'distortion', 'sev']:
            entries.append(rs[:2])
```

```python
pd.DataFrame(entries, columns=['kind', 'name'])
```

```python

```

```python
build.knowledge
```

```python
from aggregate import build
from aggregate.utilities import make_mosaic_figure, show_fig
```

```python
ans, df = build.show('B.*Roll', logger_level=30)
```

```python
act = build('agg Tw1 10 claims sev gamma 100 cv 0.5 poisson')
act
```

```python
act.plot()
```

```python
act.density_df.loc[0, 'p'], np.exp(-10)
```

```python
a = 4  # 1/cv**2; Var agg = lambda EX2= lambda beta**2 a(a+1), cv =  sqrt(a+1 / lambda alpha) 
cv = np.sqrt((a+1) / (10 * a)) 
cv
```

```python
from aggregate.utilities import tweedie_convert
```

```python
tweedie_convert(λ=10, m=100, cv=0.5) 
```

```python
# mean, p, disperson format: 
act2 = build('agg Tw2 tweedie 1000 1.2 31.3985803938698')
act2
```

```python
pd.concat((act.density_df.p, act2.density_df.p), axis=1).plot(xlim=[-20, 2000])
```

```python
welcome(rows=162)
```

```python
build.knowledge
```

```python
build
```

```python
from aggregate.utilities import pprint
build.show('^A.*', 'agg', False, False).program.apply(lambda x: pprint(x, html=True));
```

```python

```

```python
pprint?
```

```python
from aggregate import build
```

```python
# a = build('agg CAT 1.7 claims sev 1000 * pareto 1.7 - 1700 poisson', update=False, log2=18)
a = build('agg NONCAT 5 claims sev lognorm 100 cv [.4 .8 1 2] wts=4 mixed gamma .4', update=True, log2=18)
```

```python
a
```

```python
a.describe
```

```python
a.statistics_df.T
```

```python
a.statistics_total_df
```

```python
a.audit_df
```

```python
a.report_df
```

```python
a.recommend_bucket(16)
```

```python
a
```

```python
a.sev_cdf(1000000)
```

```python
1e6 / 2**18
```

```python
a.update(log2=18, bs=4)
```

```python
a
```

```python
a.audit_df
```

```python
df = a.audit_df.copy()
```

```python
df.columns = df.columns.str.split('_', n=2, expand=True)
```

```python
df
```

```python
df.replace(np.nan, '')
```

# Catastrophe Modelers

1851-2017

| Category | Count | Frequency |
|:---------|------:|----------:|
| 1        |   116 |      0.69 |
| 2        |    75 |      0.45 |
| 3        |    76 |      0.46 |
| 4        |    20 |      0.12 |
| 5        |     3 |      0.02 |

Overall severity from RMI course lognormal (19.6, 2.58)

```python
from aggregate import build
```

```python
2e10 / 2**18, np.exp(19.8 + 2.58**2/2), np.log(1e9), np.array([19.0, 19.2, 19.4, 19.6, 19.8]) - np.log(1e9)
```

```python
2**8
```

```python
import scipy.stats as ss 
```

```python
mu = -1
sig = 2.58
fz = ss.lognorm(sig, scale=np.exp(mu))
m, v = fz.stats()
m = float(m)
v = float(v)
m, v, np.exp(mu + sig**2/2), v + m**2, np.exp(2*mu + 2*sig**2), np.exp(3*mu + 9*sig**2/2)
```

```python
from scipy.integrate import quad
```

```python
argkw = dict(limit=1000, epsabs=1e-6, epsrel=1e-6, full_output=1)
a, e, m = quad(lambda x: fz.isf(x) ** 2, 2**-52, 1, **argkw)
a, e, m['last']
```

```python
a, e, *m = quadx(lambda x: fz.isf(x) ** 2, 2**-52, 1, **argkw)
```

```python
argkw = dict(limit=1000, epsabs=1e-6, epsrel=1e-8, full_output=1)
a, e1, *r1 = quad(lambda x: 3 * x**2 * fz.sf(x), 0, np.inf, **argkw)
argkw = dict(limit=1000, epsabs=1e-6, epsrel=1e-8, full_output=1)
b_basic, e, *r = quad(lambda x: fz.isf(x) ** 3, 2**-52, 1, **argkw)
b0, e, *r = quad(lambda x: fz.isf(x) ** 3, 2**-52, 1, **argkw, points=[1e-17, 1e-14, 1e-10, 1e-6, 1e-4])

# b, e, *r = quad(lambda x: fz.isf(x) ** 3, 2e-16, 1e-4, **argkw)
# c, e, *r = quad(lambda x: fz.isf(x) ** 3, 1e-4, 1, **argkw)

b, e, *r = quad(lambda x:  3 * x**2 * fz.sf(x), 0, 30, **argkw)
print(e, r[0]['last'])
c, e, *r = quad(lambda x:  3 * x**2 * fz.sf(x), 30, np.inf, **argkw)
print(e, r[0]['last'])
```

```python
pd.Series({'ex2 exact': np.exp(3*mu + 9*sig**2/2), # v+m**2, 
           'x S': a, 
           'x S err': e,
           'x S iter': r1[0]['last'],
           'isf': b_basic, 
           'isf with points': b0, 
           'isf split at 1e-4': np.array(b) + np.array(c)
          }
           ).to_frame()
           
```

```python
a, b_basic
```

```python
# actual answer for ex2: logX2 = 2 logX has 2mu 2sigma, mean exp(2mu + 2sig^2)
np.exp


```

```python
from aggregate import build
```

```python
np.exp(19.8) / 1e9
```

```python
cat = build('agg USWind [0.69 0.45 0.45 0.12 0.02] claims sev [exp(-1.7233),    exp(-1.5233),    exp(-1.3233),    exp(-1.1233),   exp(-0.92327)] * lognorm [2.18 2.28 2.38 2.45 2.58] poisson '
            'aggregate ceded to 50 x 0 and 50 x 50 and 100 x 100 and 100 x 200 and 100 x 300 and 100 x 400 and 500 x 500 and 1e4 x 1000 '
            'note{losses in billions}', 
            log2=18, bs=1/2**4, normalize=False)
```

```python
cat
```

```python
cat.reinsurance_description(), cat.reinsurance_kinds()
```

```python
with pd.option_context('display.max_rows', 10, 'display.max_columns', 15, 'display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False):
    display(cat.reins_audit_df)
```

```python
cat
```

```python
cat.plot()
```

```python
cat.q(0.99)
```

```python
cat.audit_df.T
```

```python
cat.report_df
```

# Reinsurance: check!

```python
from aggregate import build
from aggregate.utilities import make_ceder_netter
opt = ('display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False)
```

```python
reins_list = [(1.0, 10.0, 0.0), (1.0, 10.0, 10.0), (1.0, 10.0, 20.0), (1.0, 10.0, 30.0), (1.0, 10.0, 40.0), (1.0, np.inf, 50.0)]
test = build('agg TEST 5 claims dsev [1:100] poisson', bs=1, update=True)
reins_list
```

```python
test = build('agg TEST 5 claims dsev [0:100] occurrence ceded to tower [10:50:10] poisson', bs=1, update=True)

display(test)
# print(test.spec)

with pd.option_context(*opt):
    display(test.reins_audit_df)
```

```python
testa.spec
```

```python
5e6 / 2**18
```

```python
np.exp(2.5**2/2)
```

```python

```

```python

```

```python

```

```python
bahnemann = build('agg Bahn 1 claim sev 50 * lognorm 2.5 occurrence ceded to tower [100 200 500 1000 2000 3000 4000 5000] fixed', log2=18, bs=25)
with pd.option_context('display.max_rows', 100, 'display.max_columns', 15, 'display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False):
    display(bahnemann)
    display(bahnemann.reins_audit_df)

```

```python
# testo = build('agg TESTO 5 claims dsev [0:100] poisson aggregate ceded to tower [0:1000:100]', bs=1)
testo = build('agg TESTO 5 claims dsev [0:100] poisson aggregate net of 50 po 100 x 100', bs=1)
# testo = build('agg TESTO 5 claims dsev [0:100] occurrence ceded to tower [25 50 75 100] poisson', bs=1)

display(testo)

print(testo.spec['occ_reins'], \
testo.spec['agg_reins'])

with pd.option_context('display.max_rows', 100, 'display.max_columns', 15, 'display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False):
    display(testo.reins_audit_df)
    display(testo.audit_df.T)
    display(testo.report_df)
```

```python
from aggregate import build
```

```python
# testa = build('agg TESTA 5 claims dsev [0:100] poisson aggregate ceded to tower [0:1000:100]', bs=1)
testa = build('agg TESTA 5 claims dsev [0:100] poisson aggregate ceded to 100 x 50', bs=1)
# testa = build('agg TESTA 5 claims dsev [0:100] poisson aggregate ceded to tower [100 200]', bs=1, log_level=20)

display(testa)

print(testa.spec['occ_reins'], \
testa.spec['agg_reins'])

with pd.option_context('display.max_rows', 100, 'display.max_columns', 15, 'display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False):
    display(testa.reins_audit_df)
```

```python
testo.report_df.T
```

```python
testg = build('agg TESTG 5 claims dsev [0:100] poisson', bs=1, log2=10)

display(testg)

testg.plot()
```

```python
testg.q(1)
```

```python
testg.density_df.loc[100, 'lev']
```

```python
testg.density_df[['loss', 'p_total']].to_csv('\\temp\\aggloss.csv')
```

```python
fz = ss.lognorm(2.5, scale=50)
m, v, sk = fz.stats('mvs')
m, np.exp(2.5**2/2) * 50, v + m*m
```

```python
import scipy.integrate as si
```

```python
si?
```

```python
   quad          -- General purpose integration
   quad_vec      -- General purpose integration of vector-valued functions
   dblquad       -- General purpose double integration
   tplquad       -- General purpose triple integration
   nquad         -- General purpose N-D integration
   fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n
   quadrature    -- Integrate with given tolerance using Gaussian quadrature
   romberg       -- Integrate func using Romberg integration
```

```python
from aggregate import Severity, logger_level
import scipy.stats as ss
from scipy.integrate import quad, fixed_quad, quadrature, romberg
```

```python
logger_level(31)
```

```python
def test_suite(m=50, sig=2.5):
    fc = lambda x: f'{x:,.3f}'
    fg = lambda x: f'{x:.7g}'
    fp = lambda x: f'{x:.1%}'
    
    int_fun = quad
    kwargs = {'limit': 50}
    
    mu = np.log(m)

    print('actual moments f', [(n, fc(np.exp(n*mu + n**2 * sig**2/2))) for n in range(1, 4)])
    print('actual moments g', [(n, fg(np.exp(n*mu + n**2 * sig**2/2))) for n in range(1, 4)])
    n=2
    ex2 = np.exp(n*mu + n**2 * sig**2/2)
    n=3
    ex3 = np.exp(n*mu + n**2 * sig**2/2)
    
    s = Severity('lognorm', sev_a=sig, sev_scale=m)
    m, v = s.stats()
    print('Severity class mean and ex2', fc(m), fc(v + m**2))

    # direct integrals
    small = s.ppf(1e-9)
    big = s.isf(1e-9)
    print('\nsmall/big/sf(big):', fc(small), fc(big), s.sf(big))
    big *= 1
    print('\nn=2')
    a1 = int_fun(lambda x: 2 * x * s.sf(x), 0, big, **kwargs) 
    a2 = int_fun(lambda x: 2 * x * s.sf(x), big, 1e4*big, **kwargs) 
    print([fc(i) for i in a1[:2]])
    print([fc(i) for i in a2[:2]])
    print('est ex2:', fc(a1[0] + a2[0]))
    print('actual ex2:', fc(ex2))
    print('error:     ', fc(a1[0] + a2[0] - ex2), fp((a1[0] + a2[0] - ex2)/ex2))
    
    print('\nn=3')
    a3 = int_fun(lambda x: 3 * x**2 * s.sf(x), 0, big, **kwargs) 
    a4 = int_fun(lambda x: 3 * x**2 * s.sf(x), big, 1e10*big, **kwargs) 
    print([fg(i) for i in a3[:2]])
    print([fg(i) for i in a4[:2]])
    print(fg(a3[0] + a4[0]))
    print('est ex3:   ', fg(a3[0] + a4[0]))
    print('actual ex3:', fg(ex3))
    print('error:', fg(a3[0] + a4[0] - ex3), fp((a3[0] + a4[0] - ex3)/ex3))

```

```python
test_suite(m=100, sig=3.5)
```

```python
s = Severity('lognorm', sev_a=2.5, sev_scale=50)
m, v = s.stats()
m, v + m**2
```

```python
mu = np.log(50)
sig = 2.5
[(n, EngFormatter(3, False)(np.exp(n*mu + n**2 * sig**2/2))) for n in range(4)]
```

```python
s.isf(1e-15)
```

```python

```

```python
quad(lambda x: 2 * x * s.sf(x), s.isf(1e-15),  10000*s.isf(1e-15), limit=50) #, points=[s.isf(0.99), s.isf(0.01), s.isf(1e-6)])
```

```python
quad(lambda x: 3 * x**2 * s.sf(x), 0, s.isf(1e-15)), quad(lambda x: 3 * x**2 * s.sf(x), s.isf(1e-15), np.inf) #, limit=50, points=[s.isf(0.99), s.isf(0.01), s.isf(1e-6)])
```

```python
quad(lambda x: 3 * x**2 * s.sf(x), 0, 1000*s.isf(1e-15), limit=50) #, points=[s.isf(0.99), s.isf(0.01), s.isf(1e-6)])
```

```python
quad(lambda x: 3 * x**2 * s.sf(x), 0, 1000*s.isf(1e-15), points=[s.isf(0.99), s.isf(0.01), s.isf(1e-6)])
```

```python
ans = []
bp = [0, s.isf(0.99), s.isf(0.01), s.isf(1e-6), s.isf(1e-10), s.isf(1e-14), 10e10 * s.isf(1e-15)] 
for l, u in zip(bp[:-1], bp[1:]):
    ans.append((l, u, *quad(lambda x: 3 * x**2 * s.sf(x), l, u)))
```

```python
df = pd.DataFrame(ans, columns=['lower', 'upper', 'ans', 'err'])
df.loc['sum'] = df.sum()
df['re'] = df.err / df.ans
df
```

```python

```

```python
with pd.option_context('display.float_format', EngFormatter(3, True), 'display.multi_sparse', False):
    display(df)
```

```python
EngFormatter??
```

```python
from aggregate import student_build as build

```

```python
p = build('''
port test
    agg A 1 claim dsev [1:6] fixed
    agg B 1 claim dsev [1:6] poisson
    agg C 1 claim dsev [1:6] mixed gamma 0.5
''', )
```

```python
p
```

```python
28/16
```

```python
sde = build('agg SDE dfreq [1 2 3] [1/2 1/4 1/4] dsev [1:6]')

sde 

```

```python
sde.plot()
```

| Number of vehicles | Number of intervals | Probability $N$ |
|:------------------:|:-------------------:|:---------------:|
|         0          |          4          |    4/16=0.25    |
|         1          |          6          |   6/16=0.375    |
|         2          |          4          |    4/16=0.25    |
|         3          |          2          |   2/16=0.125    |
|  Total intervals   |         16          |                 |

You also observe the number of occupants per vehicle giving the **severity distribution** $X$.

| Number of occupants | Number of vehicles | Probability $X$ |
|:-------------------:|:------------------:|:---------------:|
|          1          |         10         |   10/16=0.625   |
|          2          |         3          |   3/16=0.1875   |
|          3          |         0          |        0        |
|          4          |         3          |   3/16=0.1875   |
|   Total vehicles    |         16         |                 |

The average number of occupants equals 1.75=(10+6+12)/16=28/16

```python
a = build('agg Traffic dfreq [0 1 2 3] [4/16 6/16 4/16 2/16] '
          'dsev [1 2 4] [10/16 3/16 3/16]')
```

```python
a
```

```python
20/16 * 28/16
```

```python
28/16
```

```python
a2 = build('agg Traffic.Po 1.25 claims  '
          'dsev [1 2 4] [10/16 3/16 3/16] '
          'poisson')
a2
```

```python
from aggregate.utilities import make_mosaic_figure
```

```python
a.plot()
```

```python
a2.plot()
```

```python
import scipy.stats as ss 
for i in [4.5, 3.5, 2.5, 1.5, .5]:
    p = ss.pareto(i)
    print(i, p.stats('mvsk'))
```

```python
a2.density_df.query('p_total > 0')
```

```python
f, axd = make_mosaic_figure('AB')

axd['A'].plot(a.xs, a.agg_density.cumsum())

axd['A'].plot(a2.xs, a2.agg_density.cumsum())
```
