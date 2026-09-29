You are acting as a copy editor, but one with domain expertise AND programming expertise - since you wrote a lot of this with me!

I need (a lot of) help with the docs. We need to enforce some consistency and tidy up. I need you do to the five C editor review for me, systematically. we can work ch by ch. let's start with ch 2 - the biggie. First, remember my voice and writing rules (see bottom).

Five Cs: clear (coherent/comprehesible), consistent, correct, concise, complete,

I want each ch to start with This chapter <brief overview> and each section with This section <brief overview>. Sub subs just get on with what they are doing.

Currently that is followed by an "On this page toc". But that is dumb cos sphinx makes html and pdf tocs as part of layout. So we should delete those.

some sections I wrote, some you wrote. ideally at the end they all read like mine.

Look out for excessive subsectioning. Eg the Computation Pipeline are dumps from your drafts and need Steve-izing in tone and organization.

Look out for things like "landed; dev/done" which are meta-notes with no place in the docs. Peppered in Automatic Grid Selection.

The info string contract contains tables that appear to be custom formatted. looking at the code not clear to me how that happens. I'm fine with the tables, but want all tables to be consistent.

throughout: there are unicode chars like ∈. These should all be replaced with their TeX equivalents.

Be consistent that classes etc use ``ClassName``.

Keep bolding and emphasis in the text to a minimum. Definitions can be bolded the first time they appear.

String widths in code blocks and output should be no more than 80 chars wide as far as possible (ie when we chose a width). the 20 min intro section manages this well.

agg in one, five, 20 mins: let's just go with 20 min intro. delete the other two.

as far as possible avoid "the table below/above" "the next section" etc. ALways label the section and refer to the label. that way we can move blocks of code around and it all stays comprehesible.

all decl programs should be laid out multiline mode and indented. this atm we have ( in Frequency: Poisson is an opinion)

po = build('agg Po 100 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson')

that should be

po = build('''
    agg Po
        100 claims
        1000 xs 0
        sev lognorm 100 cv 1.5
        poisson
''')

(form returned by format_program). i'm neutral on the overall indenting but it seems harmless.

lay into the SUVA-Use cycle laid out in 37114FE2, "The examples illustrate the recommended specify-update-validate-adjust-use (SUVA-use)
workflow:
• Specify the gross compound using DecL.
• Updatethenumericalapproximationusingtheupdatemethod(performedautomaticallyfor
objects created using DecL and build).
• Validate the results are “not unreasonable” by reviewing the diagnostics; if necessary, adjust
the calculation parameters and re-run update.
• Adjust the specification for reinsurance and update using the same parameters.
• Use the output.
The SUVA-use workflow leverages the built-in validation on objects without reinsurance and
helps ensure the validity of net and ceded distributions."

Rather than "working cycle is declare, build, validate, trust. ". I do like Trust but verify.

20 min: current "Everything is pandas
No export step: the distribution is a DataFrame, the reports are DataFrames, and plots are matplotlib:"  That's way too cryptic and not a complete sentence. WLS (write like steve).

---


# Hash refs to papers

Full text lives in, eg C:/S/ShardedFullText/B8/B83FDF8371_2024_Ben Rached-Hoel-Meo_fast accurate numerical method left tail sums independent random variables.pdftotext.md. You can x-ref against the uber-library.bib. the filename is HASH_YEAR_AUTHORS_title.md with some trimming.

# How to "Write like Steve"

**RULE:** No dashes, em, en, any dash. Use colons, semicolons, separate sentence.

**GUIDELINE:** Show don't tell: don't tell the reader what to do or think. Show the read and explain why it is important. They draw their own conclusions.

**RULE:** Do not start sentences with So (although I know I do).

**RULE:** No naked "this". This shows that... --> This **what** shows that....

**RULE:** Do not use the words: bite(s), load-bearing,

## Example 1

### Two ways in: ``build(x)`` and ``build.recipe(x)``
This is the distinction worth learning first, because both take the same string and return different things.

--> There is an important distinction between ``build(x)`` and ``build.recipe(x)``. Both take the same string argument but they return different things.


## Example 2

**RULE:** Do not merge the titles into the text, for example:

### ``program`` versus ``pprogram``

Every DecL-created object carries both.

--> Every DecL-created object carries both a ``program`` and ``pprogram`` attribute.

## Other Examples

| AI-Speak | Steve-Speak | Notes |
|:--|:--|:--|
| on the same rhythm | in the same way | |
|The asymmetry between the two envelopes is real and worth knowing.| There is a real asymmetry between the two envelopes. | It is implicit it is worth knowing; I would not point it out otherwise |
| The info string contract | The info string | "contract" is maybe dev jargon. we don't want it. |
| What the Underwriter is | The Underwriter | The section tells us what it is. |
| Two ways in: build(x) and build.recipe(x) | Comparing build(x) and build.recipe(x)

