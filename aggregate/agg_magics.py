from IPython.core.magic import magics_class, Magics
from IPython.core.magic import cell_magic, line_magic
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring

@magics_class
class AggregateMagic(Magics):
    """
    description: implements magics to help using Aggregate class

    """

    # @magic_arguments()
    # @argument('-d', '--debug', action='store_false', help='Debug mode; show commands.')
    @line_magic
    def agg(self, line):
        """
        Implements the agg magic. Allows::

            %agg ObName 1 claim dsev [1:6] poisson

        which is exapneded to::

            ObName = build("agg ObName 1 claim dsev [1:6] poisson")


        """

        # args = parse_argstring(self.agg, line)
        # print(args)

        debug = False
        if line.find('-d') >= 0:
            debug = True
            line = line.replace('-d', '')

        nm = line.strip().split(' ')[0]
        cmd0 = f'{nm} = build("agg {line}")'
        cmd1 = f'qd({nm})'
        if debug:
            print(cmd0)
            print(cmd1)
        self.shell.ex(cmd0)
        self.shell.ev(cmd1)

    @magic_arguments()
    @argument('--log2', type=int, default=16, help='Specify log2 to determine the number of buckets.')
    @argument('--bs', type=str, default='', help='Set bucket size, converted to float.')
    @argument('-u', '--update', action='store_false', help='Suppress update.')
    @argument('-n', '--normalize', action='store_false', help='Do not normalize severity.')
    @cell_magic
    def decl(self, line, cell=None):
        """
        WIP NYI for functional.

        """
        args = parse_argstring(self.decl, line)

        print(args)
        bs = 0.
        if args.bs != '':
            try:
                bs = eval(args.bs)
            except:  # noqa
                pass

        cmd0 = (f'a = build("""{cell}""", update={args.update}, normalize={args.normalize},'
                f'log2={args.log2}, bs={bs})')
        self.shell.ex(f'cmd0')
        self.shell.ex(f'cmd0')


def load_ipython_extension(ipython):
    ipython.register_magics(AggregateMagic)