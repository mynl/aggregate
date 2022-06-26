# debug and tester files for aggregate

import aggregate as agg
from aggregate import build


if __name__ == '__main__':

    a = build('agg Test 5 claims sev lognorm 30 cv .3 poisson')

    print(a)