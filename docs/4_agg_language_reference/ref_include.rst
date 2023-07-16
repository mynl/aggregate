answer              	::= agg_out
                    	 | port_out
                    	 | distortion_out
                    	 | expr

distortion_out      	::= DISTORTION name ID expr
                    	 | DISTORTION name ID expr "[" numberl "]"

port_out            	::= PORT name note agg_list

agg_list            	::= agg_list agg_out
                    	 | agg_out

agg_out             	::= AGG name exposures layers sev_clause occ_reins freq agg_reins note
                    	 | AGG name dfreq layers sev_clause occ_reins agg_reins note
                    	 | AGG name TWEEDIE expr expr expr note
                    	 | AGG name builtin_agg occ_reins agg_reins note
                    	 | builtin_agg agg_reins note

sev_out             	::= SEV name sev note
                    	 | SEV name dsev note

freq                	::= freq ZM expr
                    	 | freq ZT
                    	 | MIXED ID expr expr
                    	 | MIXED ID expr
                    	 | FREQ expr expr
                    	 | FREQ expr
                    	 | FREQ

agg_reins           	::= AGGREGATE NET OF reins_list
                    	 | AGGREGATE CEDED TO reins_list
                    	 |  %prec LOW

occ_reins           	::= OCCURRENCE NET OF reins_list
                    	 | OCCURRENCE CEDED TO reins_list
                    	 | 

reins_list          	::= reins_list AND reins_clause
                    	 | reins_clause
                    	 | tower

reins_clause        	::= expr XS expr
                    	 | expr SHARE_OF expr XS expr
                    	 | expr PART_OF expr XS expr

sev_clause          	::= SEV sev
                    	 | dsev
                    	 | BUILTIN_SEV

sev                 	::= sev picks
                    	 | sev "!"
                    	 | sev2 weights splice
                    	 | BUILTIN_SEV

dsev                	::= dsev "!"
                    	 | DSEV doutcomes dprobs

sev2                	::= sev1 PLUS numbers
                    	 | sev1 MINUS numbers
                    	 | sev1

sev1                	::= numbers TIMES sev0
                    	 | sev0

sev0                	::= ids numbers CV numbers
                    	 | ids numbers numbers
                    	 | ids numbers
                    	 | ids xps
                    	 | ids

xps                 	::= XPS doutcomes dprobs

dfreq               	::= DFREQ doutcomes dprobs

picks               	::= PICKS "[" numberl "]" "[" numberl "]"

doutcomes           	::= "[" numberl "]"
                    	 | "[" expr RANGE expr "]"
                    	 | "[" expr RANGE expr RANGE expr "]"

dprobs              	::= "[" numberl "]"
                    	 | 

weights             	::= WEIGHTS EQUAL_WEIGHT expr
                    	 | WEIGHTS "[" numberl "]"
                    	 | 

splice              	::= SPLICE "[" numberl "]"
                    	 | 

layers              	::= numbers XS numbers
                    	 | tower
                    	 | 

tower               	::= TOWER doutcomes

note                	::= NOTE
                    	 |  %prec LOW

exposures           	::= numbers CLAIMS
                    	 | numbers LOSS
                    	 | numbers PREMIUM AT numbers LR
                    	 | numbers EXPOSURE AT numbers RATE

ids                 	::= "[" idl "]"
                    	 | ID

idl                 	::= idl ID
                    	 | ID

builtin_agg         	::= expr INHOMOG_MULTIPLY builtin_agg
                    	 | expr TIMES builtin_agg
                    	 | builtin_agg PLUS expr
                    	 | builtin_agg MINUS expr
                    	 | BUILTIN_AGG

name                	::= ID

numbers             	::= "[" numberl "]"
                    	 | "[" expr RANGE expr "]"
                    	 | "[" expr RANGE expr RANGE expr "]"
                    	 | expr

numberl             	::= numberl expr
                    	 | expr

expr                	::= atom

atom                	::= atom DIVIDE atom
                    	 | "(" atom ")"
                    	 | EXP atom
                    	 | atom EXPONENT atom
                    	 | NUMBER

FREQ                    ::= 'binomial|poisson|bernoulli|pascal|geometric|neymana?|fixed|logarithmic|negbin'

BUILTINID               ::= 'sev|agg|port|meta.ID'

NOTE                    ::= 'note{TEXT}'

EQUAL_WEIGHT            ::= "="

AGG                     ::= 'agg'

AGGREGATE               ::= 'aggregate'

AND                     ::= 'and'

AT                      ::= 'at'

CEDED                   ::= 'ceded'

CLAIMS                  ::= 'claims|claim'

CONSTANT                ::= 'constant'

CV                      ::= 'cv'

DFREQ                   ::= 'dfreq'

DSEV                    ::= 'dsev'

EXP                     ::= 'exp'

EXPONENT                ::= '^|**'

INHOMOG_MULTIPLY        ::= "@"

INFINITY                ::= 'inf|unlim|unlimited'

LOSS                    ::= 'loss'

LR                      ::= 'lr'

MIXED                   ::= 'mixed'

NET                     ::= 'net'

OCCURRENCE              ::= 'occurrence'

OF                      ::= 'of'

PART_OF                 ::= 'po'

PERCENT                 ::= '%'

PORT                    ::= 'port'

PREMIUM                 ::= 'premium|prem'

SEV                     ::= 'sev'

SHARE_OF                ::= 'so'

TO                      ::= 'to'

WEIGHTS                 ::= 'wts|wt'

XPS                     ::= 'xps'

XS                      ::= "xs|x"

