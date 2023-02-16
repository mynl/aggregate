

.. _2_x_fft_convolution:

Numerical Methods and FFT Convolution
=======================================

**Objectives:**  Describe the numerical distribution representation and FFT convolution algorithm that underlie all computations in ``aggregate``.

**Audience:** User who wants a detailed understanding of the computational algorithms, potential errors, options, and parameters.

**Prerequisites:** Probability theory and aggregate distributions; complex numbers and matrix multiplication; numerical analysis, especially numerical integration; basics of Fourier transforms and series helpful.

**See also:**  :doc:`../2_User_Guides`.

**Contents:**

* :ref:`num hr`
* :ref:`num overview`
* :ref:`num how agg reps a dist`
* :ref:`num ft convo`
* :ref:`num floats`

.. _num hr:

Helpful References
--------------------

Actuarial and operational risk books and papers

* :cite:t:`Gerber1982`
* :cite:t:`Buhlmann1984`
* :cite:t:`Embrechts1993`
* :cite:t:`WangS1998`
* :cite:t:`Grubel1999`
* :cite:t:`Mildenhall2005a`
* :cite:t:`Schaller2008`
* :cite:t:`Kaas2008`
* :cite:t:`Embrechts2009a`
* :cite:t:`Shevchenko2010`

Books on probability covering characteristic functions,
:math:`t\mapsto \mathsf E[e^{itX}]`

* :cite:t:`Loeve1955`
* :cite:t:`feller71`
* :cite:t:`Lukacs1970bk`
* :cite:t:`billingsley`
* :cite:t:`Malliavin1995bk`
* :cite:t:`McKean2014bk`

Books on Fourier analysis and Fourier transforms,
:math:`t\mapsto \mathsf E[e^{-2\pi itX}]`, the same concept with slightly different notation. Malliavin is a sophisticated treatment of both Fourier analysis and probability.

* :cite:t:`Stein1971bk`
* :cite:t:`Stein2011bk`
* :cite:t:`Strang1986am`
* :cite:t:`Terras2013`
* :cite:t:`Korner2022`


.. include:: 5_x_nm_overview.rst
.. include:: 5_x_nm_discrete_rep.rst
.. include:: 5_x_nm_ft_conv_algo.rst
.. include:: 5_x_nm_error_analysis_params.rst
.. include:: 5_x_nm_other_algos.rst
.. include:: 5_x_nm_floats.rst
