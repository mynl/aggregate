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

```python

```
