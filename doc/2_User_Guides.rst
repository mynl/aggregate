.. 2022-11-10: reviewed

**************
User Guides
**************

The User Guides show how to **access** ``aggregate`` functionality
and **apply** it to solve various actuarial problems. It alternates between
access-oriented reference guides and application-oriented problem guides. New
users should start with the :doc:`2_user_guides/2_x_student` guide.
See :doc:`5_Technical_Guides` for the theory and implementation details.

#. :doc:`2_user_guides/2_x_10mins`: A whirlwind tour---don't expect to understand everything the first time! Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a *long* 10 minutes.

#. :doc:`2_user_guides/2_x_student` (problem): Introduction to aggregate distributions using simple discrete examples for actuarial science majors and STAM, MAS-I, CS-2 candidates.

#. :doc:`2_user_guides/2_x_actuary_student` (problem): Introduction to aggregate distributions in actuarial language using more realistic insurance examples for actuarial analysts and CAS Part 8 candidates.

#. :doc:`2_user_guides/2_x_underwriter` (reference): Working with the :class:`Underwriter` object.

#. :doc:`2_user_guides/2_x_dec_language` (reference): Specification of the Dec language (DecL) used to specify aggregate distributions using familiar insurance terminology.

#. :doc:`2_user_guides/2_x_aggregate` (reference): Specifying, creating and working with the :class:`Aggregate` object.

#. :doc:`2_user_guides/2_x_ir_pricing` (problem): Applications of the :class:`Aggregate` class to individual risk pricing, including LEVs, ILFs, layering, and the aggregate insurance charge (Table L, M), illustrated using problems from CAS Part 8.

#. :doc:`2_user_guides/2_x_re_pricing` (problem):  Applications of the :class:`Aggregate` class to reinsurance exposure rating, including swings and slides, aggregate stop loss and swing rated programs, illustrated using problems from CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_reserving` (problem):  Applications of the :class:`Aggregate` class to reserving, including models of loss emergence and determining ranges for IBNR and case reserves.

#. :doc:`2_user_guides/2_x_cat` (problem): Applications of the :class:`Aggregate` class to catastrophe risk evaluation and pricing using thick-tailed Poisson Pareto and lognormal models, including occurrence and aggregate PMLs and layer loss costs. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_distortion` (reference): Workings of the :class:`Distortion` class.

#. :doc:`2_user_guides/2_x_portfolio` (reference): Specifying, creating and working with the :class:`Portfolio` class, including DecL options.

#. :doc:`2_user_guides/2_x_capital` (problem): Application of the :class:`Portfolio` class to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_strategy` (problem): Application of the :class:`Portfolio` and  and :class:`Distortion` classes to strategy and portfolio management, including margin (capital) allocation, determining benchmark pricing within a portfolio using alternative pricing methodologies, and the evaluation of reinsurance.

#. :doc:`2_user_guides/2_x_case_studies` (problem): Using ``aggregate`` to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build your cases.

#. :doc:`2_user_guides/2_x_samples_rearrangement` (reference): How to build a :class:`Portfolio` from a sample; using the Iman-Conover method to induce correlation and use the rearrangement algorithm to determine VaR bounds.

#. :doc:`2_user_guides/2_x_problems` (problem):  ``aggregate`` solutions to a wide selection of problems and examples from books (Loss Models, Loss Data Analytics, ), study notes (), and academic papers ().

Guides marked *problem* are application-based; they give possible driving destinations. *Reference* guides are access-based; they describe how to unlock the car, start the engine, and engage a gear. Or, in a cooking analogy, *problems* are recipes, and *references* describe cooking techniques such as broiling or baking.

.. toctree::
    :maxdepth: 1
    :hidden:

    2_user_guides/2_x_10mins
    2_user_guides/2_x_student
    2_user_guides/2_x_actuary_student
    2_user_guides/2_x_underwriter
    2_user_guides/2_x_dec_language
    2_user_guides/2_x_aggregate
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
    2_user_guides/2_x_problems

