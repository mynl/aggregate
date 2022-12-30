.. _2_x_dec_language:

.. reviewed 2022-12-24

The Dec Language
======================

**Objectives:** Introduce the Dec Language (DecL) used to specify aggregate distributions in familiar insurance terminology.

**Audience:** User who wants to use DecL to build realistic aggregates.

**Prerequisites:** Familiar with using ``build``. Probability theory behind aggregate distributions. Insurance and reinsurance terminology.

**See also:** :doc:`2_x_re_pricing`, and :doc:`../4_dec_Language_Reference`.

**Notation:** ``<item>`` denotes an optional term.
See the note :ref:`10 mins formatting` for important information about how DecL programs are formatted and laid out in the help.

**Contents:**

.. toctree::
    :maxdepth: 4

    DecL/010_Aggregate
    DecL/020_exposure
    DecL/030_limits
    DecL/040_severity
    DecL/050_frequency
    DecL/060_mixed_severity
    DecL/065_limit_profiles
    DecL/070_vectorization
    DecL/080_reinsurance
    DecL/090_notes
    DecL/100_tweedie



Summary of Objects Created by DecL
-------------------------------------

Objects created by :meth:`build` in the DecL guide.

.. ipython:: python
    :okwarning:
    :okexcept:

    from aggregate import pprint_ex
    for n, r in build.qshow('^DecL:').iterrows():
        pprint_ex(r.program, split=20)
