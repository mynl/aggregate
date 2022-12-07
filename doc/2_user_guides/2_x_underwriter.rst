.. _2_x_underwriter:

The :class:`Underwriter` Class
===============================

**Objectives:**

**Audience:**

**Prerequisites:**

**See also:**

The Underwriter is an interface into the computational functionality of aggregate. It can

* Maintain a libraries of severity curves.
* Maintain a libraries of aggregate distributions, e.g., industry losses in major classes of business, total catastrophe losses from major perils, and other useful constructs.
* Maintain a libraries of portfolios, e.g., Bodoff's examples and Pricing Insurance Risk case studies.

The libraries are collectively called the **knowledge** of the underwriter. The knowledge is a dataframe, indexed by kind (severity, aggregate, portfolio) and name (label given on creation) and containing the aggregate program and a parsed dictionary that can be passed as keyword arguments to the appropriate ``aggregate`` class to create the object.

.. ipython:: python
    :okwarning:

    from aggregate import build
    build.knowledge.head()



A given row in the knowledge can be accessed using

.. ipython:: python
    :okwarning:

    build['A.Traffic']

and created as a Python object using

.. ipython:: python
    :okwarning:

    a = build('A.Traffic')
    a

The Underwriter class interprets DecL programs. The scripting language allows severities, aggregates and portfolios to be created using standard insurance language, see XXXX.



How :class:`Underwriter` works
---------------------------------

Each object two properties (kind, name) and three manifestations (spec, program, ob[ject])

1. kind: agg, sev, port, distortion
2. name of the object
3. spec: dictionary specification
4. program: text string, the aggregate program
5. object: the actual Python object, an instance of a class

``Underwriter._knowledge`` is a Pandas dataframe with row index (kind, name) mapping to (spec, program)

``.build`` wraps ``.write``

* calls write with update=False; it handles update with good defaults
* if only one output, strips it out of the dict and returns the object
* if only one port output, returns that

``.write``

* lowest level update function
* calls ``interpret_program``
* calls ``factory``
* reads program, preprocess, parse by line, expand sev.name, agg.name, create, update
* returns a list of Answer objects with  keys kind, name, spec, program, and object

``.write_file``

* Reads a file and passes to ``write``.

``.interpret_program`` (called by ``write``)

* maps the programs and specs together in an Answer(kind, name, spec, program, object=None)
* adds data to _knowledge

``.factory``

* Answer --> Answer with object created  the object and updated

``.interpreter_xxx``

* run programs through parser, for debugging purposes
* nothing created
* ``_interpreter_work`` does the actual parsing
* ``interpreter_line`` calls work on one line
* ``interpreter_file`` calls work on each line in a file
* ``interpreter_list`` calls work on each item in a list

