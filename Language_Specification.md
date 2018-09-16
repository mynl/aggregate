# agg Language Specification

```
ans 		name exposures layers sevs freq
			name builtinagg

name 		ID   # name for portfolio

# id list
idl 		ID
			idl ID

ids 		ID
			[ idl ]

numbers 	NUMBER
			[ numberl ]

numberl 	NUMBER
			numberl NUMBER

builtin_agg	uw.ID
			NUMBER TIMES builtin_agg
			builtin_agg TIMES NUMBER

exposures 	numbers LOSS
			numbers CLAIMS
			numbers PREMIUM AT numbers
			numbers PREMIUM AT numbers LR
			numbers PREMIUM numbers LR
			empty

layers		numbers XS numbers
			empty

# singular sev term...is a list ; in agg it is called agg_sev ==> do not use sevs
sev 		builtins
			ids numbers numbers WT weights
			ids numbers CV numbers WT weights
			numbers * sev
			sev * numbers
			sev + numbers

weights 	numbers
			empty

freq 		POISSON
			FIXED   # number must be defined in expos term!
			ID NUMBER
			ID NUMBER NUMBER
			empty


```



| A  | B  |  C |
|:---|:---|---:|
| 12 | 13 | 15 |
| 1  | 2  |  3 |

| name        | expos                 | limit                    | sev                                               | freq              |
|:------------|:----------------------|:-------------------------|:--------------------------------------------------|:------------------|
| big_mixture | 50 claims             | [50, 100, 150, 200] xs 0 | on lognorm 12 cv [1,2,3,4] wts [0.25 .25 .25 .25] | poisson           |
| A1          | 500 premium at 0.5    |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A1          | 500 premium at 0.5 lr |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A2          | 50  claims            | 30 xs 10                 | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A3          | 50  claims            |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A4          | 50  claims            | 30 xs 20                 | on gamma 12 cv .30                                | mixed gamma 0.14  |
| hcmp        | 1000 * uw.cmp         |                          |                                                   |                   |
| incmp       | uw.cmp * 0.001        |                          |                                                   |                   |
