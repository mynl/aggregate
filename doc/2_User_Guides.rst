.. 2022-11-10: reviewed
.. 2022-12-24: reviewed

**************
User Guides
**************

The User Guides show how to **access** ``aggregate`` functionality
and **apply** it to solve actuarial problems. It alternates between
access-oriented reference guides and problem and application based  practice
guides. New users should start reading the :doc:`2_user_guides/2_x_student`
or :doc:`2_user_guides/2_x_actuary_student` guide and scan
through :doc:`2_user_guides/2_x_10mins`. See :doc:`5_Technical_Guides` for
the theory and implementation details. Sections in the guides marked **Details** can be skipped.

#. :doc:`2_user_guides/2_x_10mins` (reference): A whirlwind introduction---don't expect to understand everything the first time! Basic functionality of important classes. Read in parallel with the :doc:`Student<2_user_guides/2_x_student>` or :doc:`Actuarial Student<2_user_guides/2_x_actuary_student>` practice guides. Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a *long* 10 minutes.

#. :doc:`2_user_guides/2_x_student` (practice): Introduction to aggregate distributions using simple discrete examples for actuarial science majors and STAM, MAS-I, CS-2 candidates.

#. :doc:`2_user_guides/2_x_actuary_student` (practice): Introduction to aggregate distributions in actuarial language using more realistic insurance examples for actuarial exam candidates and working actuarial analysts.

#. :doc:`2_user_guides/2_x_dec_language` (reference): Specification of the Dec Language (DecL) used to specify aggregate distributions using familiar insurance terminology.

#. :doc:`2_user_guides/2_x_ir_pricing` (practice): Applications of the :class:`Aggregate` class to individual risk pricing, including LEVs, ILFs, layering, and the aggregate insurance charge (Table L, M), illustrated using problems from CAS Part 8.

#. :doc:`2_user_guides/2_x_re_pricing` (practice):  Applications of the :class:`Aggregate` class to reinsurance exposure rating, including swings and slides, aggregate stop loss and swing rated programs, illustrated using problems from CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_reserving` (practice):  Applications of the :class:`Aggregate` class to reserving, including models of loss emergence and determining ranges for IBNR and case reserves.

#. :doc:`2_user_guides/2_x_cat` (practice): Applications of the :class:`Aggregate` class to catastrophe risk evaluation and pricing using thick-tailed Poisson Pareto and lognormal models, including occurrence and aggregate PMLs and layer loss costs. Covers material on CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_capital` (practice): Application of the :class:`Portfolio` class to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_strategy` (practice): Application of the :class:`Portfolio` and  and :class:`Distortion` classes to strategy and portfolio management, including margin (capital) allocation, determining benchmark pricing within a portfolio using alternative pricing methodologies, and the evaluation of reinsurance.

#. :doc:`2_user_guides/2_x_case_studies` (practice): Using ``aggregate`` to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build similar exhibits for your own cases.

#. :doc:`2_user_guides/2_x_samples_rearrangement` (reference): How to build a :class:`Portfolio` from a sample. Using the Iman-Conover method to induce correlation. Using the rearrangement algorithm to determine VaR bounds.

#. :doc:`2_user_guides/2_x_problems` (practice):  ``aggregate`` solutions to a wide selection of problems and examples from books (Loss Models, Loss Data Analytics), actaruial exam study notes, and academic papers.

Guides marked *practice* are problem and application based; they give possible driving destinations. *Reference* guides are access-based; they describe how to unlock the car, start the engine, and engage a gear.

.. toctree::
    :maxdepth: 1
    :hidden:

    2_user_guides/2_x_10mins
    2_user_guides/2_x_student
    2_user_guides/2_x_actuary_student
    2_user_guides/2_x_dec_language
    2_user_guides/2_x_ir_pricing
    2_user_guides/2_x_re_pricing
    2_user_guides/2_x_reserving
    2_user_guides/2_x_cat
    2_user_guides/2_x_capital
    2_user_guides/2_x_strategy
    2_user_guides/2_x_case_studies
    2_user_guides/2_x_samples_rearrangement
    2_user_guides/2_x_problems

