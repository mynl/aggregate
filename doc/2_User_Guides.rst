.. 2022-11-10: reviewed

**************
User Guides
**************

The User Guides show how to **access** ``aggregate`` functionality and **apply** it to solve various actuarial problems. It alternates between application problem-based guides and access reference guides. New users should start with the :doc:`2_user_guides/2_x_student` guide. See :doc:`../5_Techncial_Guides` for the theory and implementation details.

#. :doc:`2_user_guides/2_x_10mins`: A whirlwind tour---don't expect to understand everything the first time! Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a *long* 10 minutes.

#. :doc:`2_user_guides/2_x_student` (problem): Introduction to aggregate distributions using simple discrete examples for actuarial science majors and STAM, MAS-I, CS-2 candidates.

#. :doc:`2_user_guides/2_x_actuary_student` (problem): Introduction to aggregate distributions in actuarial language using more realistic insurance examples for actuarial analysts and CAS Part 8 candidates.

#. :doc:`2_user_guides/2_x_underwriter` (reference): Working with the :class:`Underwriter` object.

#. :doc:`2_user_guides/2_x_numerical_methods` (reference): Aggregates as a parametric class; parameters(--> problem of specifying); FFT convolution; discretization choices (switches introduced...for next section.)

#. :doc:`2_user_guides/2_x_aggregate` (reference): Specifying, creating and working with the :class:`Aggregate` object.

#. :doc:`2_user_guides/2_x_dec_language` (reference): Specification of the Dec language (DecL) to build aggregate distributions using familiar insurance terminology. The next reference guides expand on different parts of language specific to creating aggregate distributions.

   #. :doc:`2_user_guides/2_x_frequency` (reference): Description of available frequency distributions.

   #. :doc:`2_user_guides/2_x_severity` (reference): Compendium of available severity distributions.

   #. :doc:`2_user_guides/2_x_exposure` (reference): Specifying exposure, the volume of risk modeled.

   #. :doc:`2_user_guides/2_x_mixtures` (reference): Using mixed severity distributions.

   #. :doc:`2_user_guides/2_x_limits` (reference): Applying limits and deductibles.

   #. :doc:`2_user_guides/2_x_vectorization` (reference): Multiple exposures, limits, and severity curves. Limits profiles. The interaction of mixed severity and limit profiles.

   #. :doc:`2_user_guides/2_x_reinsurance` (reference): Applying occurrence and aggregate reinsurance.

   #. :doc:`2_user_guides/2_x_tweedie` (reference): Working with  Tweedie distributions as used in GLMs.

   The next problem guides give applications of the :class:`Aggregate` class:

   9. :doc:`2_user_guides/2_x_ir_pricing` (problem): Applications to individual risk pricing, including LEVs, ILFs, layering, and the aggregate insurance charge (Table L, M), illustrated using problems from CAS Part 8.

   #. :doc:`2_user_guides/2_x_re_pricing` (problem):  Applications to reinsurance exposure rating, including swings and slides, aggregate stop loss and swing rated programs, illustrated using problems from CAS Parts 8 and 9.

   #. :doc:`2_user_guides/2_x_reserving` (problem):  Applications to reserving, including models of loss emergence and determining ranges for IBNR and case reserves.

   #. :doc:`2_user_guides/2_x_cat` (problem): Applications to catastrophe risk evaluation and pricing using thick-tailed Poisson Pareto and lognormal models, including occurrence and aggregate PMLs and layer loss costs. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_distortion` (reference): Workings of the :class:`Distortion` class.

#. :doc:`2_user_guides/2_x_portfolio` (reference): Specifying, creating and working with the :class:`Portfolio` class, including DecL options. The next guides give applications of the :class:`Portfolio` and :class:`Distortion`classes.

   #. :doc:`2_user_guides/2_x_capital` (problem): Applications to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9.

   #. :doc:`2_user_guides/2_x_strategy` (problem): Applications to strategy and portfolio management, including margin (capital) allocation, determining benchmark pricing within a portfolio using alternative pricing methodologies, and the evaluation of reinsurance.

   #. :doc:`2_user_guides/2_x_case_studies` (problem): How to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build your cases.

   #. :doc:`2_user_guides/2_x_samples_rearrangement` (reference): How to build a :class:`Portfolio` from a sample; using the Iman-Conover method to induce correlation and use the rearrangement algorithm to determine VaR bounds.

#. :doc:`2_user_guides/2_x_approximation_error` (reference): How ``aggregate`` approximates a distribution and advice on selecting discretization parameters. For use when the defaults fail.

#. :doc:`2_user_guides/2_x_problems` (problem):  Provides the ``aggregate`` solution to a selection of problems and examples from books (Loss Models, Loss Data Analytics, ), study notes (), and academic papers.

Guides marked *problem* are possible driving destinations, whereas *reference* guides describe how to unlock the car, start the engine, and engage a gear. Or, in a cooking analogy, *problems* are recipes, and *references* describe cooking techniques such as broiling or baking.

.. <iframe width="1100" height="619" src="https://www.youtube.com/embed/GFP4WgHXqic" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. toctree::
    :maxdepth: 1
    :hidden:

    2_user_guides/2_x_10mins
    2_user_guides/2_x_student
    2_user_guides/2_x_actuary_student
    2_user_guides/2_x_underwriter
    2_user_guides/2_x_numerical_methods
    2_user_guides/2_x_aggregate
    2_user_guides/2_x_dec_language
    2_user_guides/2_x_frequency
    2_user_guides/2_x_severity
    2_user_guides/2_x_exposure
    2_user_guides/2_x_mixtures
    2_user_guides/2_x_limits
    2_user_guides/2_x_vectorization
    2_user_guides/2_x_reinsurance
    2_user_guides/2_x_tweedie
    2_user_guides/2_x_ir_pricing
    2_user_guides/2_x_re_pricing
    2_user_guides/2_x_reserving
    2_user_guides/2_x_cat
    2_user_guides/2_x_distortion
    2_user_guides/2_x_portfolio
    2_user_guides/2_x_capital
    2_user_guides/2_x_strategy
    2_user_guides/2_x_case_studies
    2_user_guides/2_x_samples_rearrangement
    2_user_guides/2_x_approximation_error
    2_user_guides/2_x_problems
    2_user_guides/2_x_unused

