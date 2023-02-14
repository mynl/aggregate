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
the theory and implementation details. Sections in the guides marked **Details** can be skipped. There is some duplication between sections to make them independent.

#. :doc:`2_user_guides/2_x_student` (practice): Introduction to aggregate distributions using simple discrete examples for actuarial science majors and short-term actuarial modeling exam candidates; get started using :mod:`aggregate`.

#. :doc:`2_user_guides/2_x_actuary_student` (practice): Introduce the ``aggregate`` library for working with aggregate probability distributions in the context of actuarial society exams (`SOA exam STAM <https://www.soa.org/education/exam-req/edu-exam-stam-detail/>`_, `CAS exam MAS I <https://www.casact.org/exam/exam-mas-i-modern-actuarial-statistics-i>`_, or `IFOA CS-2 <https://www.actuaries.org.uk/curriculum_entity/curriculum_entity/8>`_) and university courses in (short-term) actuarial modeling.

#. :doc:`2_user_guides/2_x_10mins` (reference): A whirlwind introduction---don't expect to understand everything the first time, but you will see what you can achieve with the package. Read in parallel with the :doc:`Student<2_user_guides/2_x_student>` or :doc:`Actuarial Student<2_user_guides/2_x_actuary_student>` practice guides. Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a *long* 10 minutes.

#. :doc:`2_user_guides/2_x_dec_language` (reference): Introduce the Dec Language (DecL) used to specify aggregate distributions in familiar insurance terminology.

#. :doc:`2_user_guides/2_x_ir_pricing` (practice): Applications of the :class:`Aggregate` class to individual risk pricing, including LEVs, ILFs, layering, and the insurance charge and savings (Table L, M), illustrated using problems from CAS Part 8.

#. :doc:`2_user_guides/2_x_re_pricing` (practice):  Applications of the :class:`Aggregate` class to reinsurance exposure rating, including swings and slides, aggregate stop loss and swing rated programs, illustrated using problems from CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_reserving` (practice, placeholder):  Applications of the :class:`Aggregate` class to reserving, including models of loss emergence and determining ranges for IBNR and case reserves.

#. :doc:`2_user_guides/2_x_cat` (practice): Applications of the :class:`Aggregate` class to catastrophe risk evaluation and pricing using thick-tailed Poisson Pareto and lognormal models, including occurrence and aggregate PMLs (OEP, AEP) and layer loss costs. Covers material on CAS Parts 8 and 9.

#. :doc:`2_user_guides/2_x_capital` (practice, placeholder): Application of the :class:`Portfolio` class to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9.

#. :doc:`2_user_guides/2_x_strategy` (practice, placeholder): Application of the :class:`Portfolio` and  and :class:`Distortion` classes to strategy and portfolio management, including margin (capital) allocation, determining benchmark pricing within a portfolio using alternative pricing methodologies, and the evaluation of reinsurance.

#. :doc:`2_user_guides/2_x_case_studies` (practice): Using :mod:`aggregate` to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build similar exhibits for your own cases.

#. :doc:`2_user_guides/2_x_samples` (reference): How to sample from :mod:`aggregate` and how to a build a :class:`Portfolio` from a sample. Inducing correlation in a sample using the Iman-Conover algorithm and determining the worst-VaR rearrangement using the rearrangement algorithm.

#. :doc:`2_user_guides/2_x_problems` (practice):  :mod:`aggregate` solutions to a wide selection of problems and examples from books (Loss Models, Loss Data Analytics), actuarial exam study notes, and academic papers. Demonstrates the method of solution and verifies the correctness of :mod:`aggregate` calculations.

Guides marked **practice** are problem and application based and give possible driving destinations; those marked **reference** are access-based and describe how to unlock the car, start the engine, and engage a gear.

Guides marked **placeholder** are work in progress, often just a sketch of planned content.

.. to complete: IR, samples,

.. placeholders are: reserving, capital, strategy

.. toctree::
    :maxdepth: 1
    :hidden:

    2_user_guides/2_x_student
    2_user_guides/2_x_actuary_student
    2_user_guides/2_x_10mins
    2_user_guides/2_x_dec_language
    2_user_guides/2_x_ir_pricing
    2_user_guides/2_x_re_pricing
    2_user_guides/2_x_reserving
    2_user_guides/2_x_cat
    2_user_guides/2_x_capital
    2_user_guides/2_x_strategy
    2_user_guides/2_x_case_studies
    2_user_guides/2_x_samples
    2_user_guides/2_x_problems

