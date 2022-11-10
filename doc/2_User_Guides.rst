.. 2022-11-10: reviewed

**************
User Guides
**************

The User Guides show how ``aggregate`` can solve various actuarial problems. It alternates between problem-based guides and reference guides. New users should start with the :doc:`2_x_student` guide.

.. Each is in a separate short document. Many of the guides include a video presentation.

.. 17 files

#. :doc:`2_user_guides/2_x_10mins`: A lightning tour---don't expect to understand everything the first time! Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a long 10 minutes.
#. :doc:`2_user_guides/2_x_student` (problem): Introduction to aggregate distributions using simple discrete examples for actuarial science majors and STAM, MAS-I, CS-2 candidates.
#. :doc:`2_user_guides/2_x_actuary_student` (problem): Introduction to aggregate distributions in actuarial language using more realistic insurance examples for actuarial analysts and CAS Part 8 candidates.
#. :doc:`2_user_guides/2_x_underwriter` (reference): Workings of the :class:`Underwriter` object.

#. :doc:`2_user_guides/2_x_aggregate` (reference): Workings of the :class:`Aggregate` object.
#. :doc:`2_user_guides/2_x_exposure` (reference): Specifying exposure.
#. :doc:`2_user_guides/2_x_mixtures` (reference): Using mixed severity distributions.
#. :doc:`2_user_guides/2_x_limits_and_mixtures` (reference): Limits and limit profiles and their interaction with mixed severities.

#. :doc:`2_user_guides/2_x_agg_language` (reference): Specification of the ``agg`` language to build aggregate distributions using familiar insurance terminology.
#. :doc:`2_user_guides/2_x_tweedie` (reference): Working with  Tweedie distributions as used in GLMs.
#. :doc:`2_user_guides/2_x_ir_pricing` (problem): Applications to individual risk pricing, including LEVs, ILFs, layering, and the aggregate insurance charge (Table L, M), illustrated using problems from CAS Part 8.
#. :doc:`2_user_guides/2_x_re_pricing` (problem):  Applications to reinsurance exposure rating, including swings and slides, aggregate stop loss and swing rated programs, illustrated using problems from CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_reserving` (problem):  Applications to reserving, including models of loss emergence and determining ranges for IBNR and case reserves.
#. :doc:`2_user_guides/2_x_cat` (problem): Applications to catastrophe risk evaluation and pricing using thick-tailed Poisson Pareto and lognormal models, including occurrence and aggregate PMLs and layer loss costs. Covers material on CAS Part 9.
#. :doc:`2_user_guides/2_x_portfolio` (reference): Workings of the :class:`Portfolio` object.
#. :doc:`2_user_guides/2_x_capital` (problem): Applications to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_samples_rearrangement` (reference): How to build a :class:`Portfolio` from a sample; using the Iman-Conover method to induce correlation and the rearrangement algorithm to determine VaR bounds.
#. :doc:`2_user_guides/2_x_distortion` (reference): Workings of the :class:`Distortion` object.
#. :doc:`2_user_guides/2_x_strategy` (problem): Applications to strategy and portfolio management, including margin (capital) allocation, determining benchmark pricing within a portfolio using alternative pricing methodologies, and the evaluation of reinsurance.
#. :doc:`2_user_guides/2_x_case_studies`: How to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build your cases.


Guides marked *problem* are possible driving destinations, whereas *reference* guides describe how to unlock the car, start the engine, and engage a gear. Or, in a cooking analogy, *problems* are recipes, and *references* describe cooking techniques such as broiling or baking.

.. Table M and Table L!
.. https://www.wcirb.com/content/california-retrospective-rating-plan
.. ISO Retro Rating Plan
.. Fisher et al case study spreadsheet...

.. <iframe width="1100" height="619" src="https://www.youtube.com/embed/GFP4WgHXqic" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



.. Guides
   -------

.. toctree::
    :maxdepth: 1
    :hidden:

    2_user_guides/2_x_10mins
    2_user_guides/2_x_student
    2_user_guides/2_x_actuary_student
    2_user_guides/2_x_underwriter
    2_user_guides/2_x_aggregate
    2_user_guides/2_x_exposure
    2_user_guides/2_x_mixtures
    2_user_guides/2_x_limits_and_mixtures
    2_user_guides/2_x_agg_language
    2_user_guides/2_x_tweedie
    2_user_guides/2_x_ir_pricing
    2_user_guides/2_x_re_pricing
    2_user_guides/2_x_reserving
    2_user_guides/2_x_cat
    2_user_guides/2_x_portfolio
    2_user_guides/2_x_capital
    2_user_guides/2_x_samples_rearrangement
    2_user_guides/2_x_distortion
    2_user_guides/2_x_strategy
    2_user_guides/2_x_case_studies
    2_user_guides/2_x_unused

