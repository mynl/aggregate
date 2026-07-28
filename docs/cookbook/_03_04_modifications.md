<!-- _03_04_modifications.qmd — Frequency: modifications -->

## Modified frequencies {#sec-frequency-modified}

*This is a recipe for* zero truncated and zero modified frequencies.

```{python}
#| echo: false
from _setup import *
from textwrap import fill
```

**Beat 1 — the DecL.**

```{python}
#| echo: false
egs = """4 claims dsev[1] poisson zm 0.5      note{base 4      → E[N] = 2.0373}
4 claims dsev[1] poisson zm 0.5 !    note{base 7.9973 → E[N] = 4}
4 claims dsev[1] poisson zt          note{base 4      → E[N] = 4.0746}
0.5 claims dsev[1] poisson zt        note{base 0.5    → E[N] = 1.2707}""".split('\n')
for eg in egs:
    print(eg)
```

**Beats 2 & 3 — build, validate, exhibits.**

```{python}
#| echo: false
for i, eg in enumerate(egs):
    print(s:=f'Example {i+1}', '=' * len(s), sep='\n')
    a = build(f'agg TEST{i+1} {eg}')
    print(a.pprogram)
    print()
    print(fill(a._text_info_blob(), 65))
    print()
    qd(a.summary_df.fillna(''))
    print()
    print(f'Note: {a.note}')
    print('-' * 80)
    print()
```

<!-- TODO(author): show the count pmf and the resulting aggregate for each. -->

**Beat 4 — the check.** [Check-Independent-Oracle]

<!-- TODO(author): each count's mean/variance matches its closed form. -->
