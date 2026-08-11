## Specify: declaration in, object out

Specifying costs nothing and computes nothing. It turns text into a live object with a spec, analytic moments, and no distribution yet.

The knowledge base is why `build('MyBook')` works as well as `build('agg MyBook ...')`. Text is parsed once into a recipe, the recipe is remembered by name, and building is a separate step from parsing.

```{mermaid}
%%| label: fig-specify
%%| fig-cap: "Parsing, remembering and building are three steps, not one."
flowchart TD
    decl["DecL program<br/>agg MyBook 100 claims sev lognorm 100 cv 2"]
    lex(["Lark parser<br/>decl.lark grammar"])
    recipe["Recipe<br/>kind, name, spec dict"]
    kb[("knowledge base<br/>named programs")]
    obj["live object<br/>Aggregate, Portfolio,<br/>Severity, Distortion"]
    eng(["engine: update()<br/>FFT convolution on a chosen grid"])
    frames[("the frames")]

    decl --> lex
    lex --> recipe
    recipe --> kb
    kb --> recipe
    recipe --> obj
    obj --> eng
    eng --> frames

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    class decl,recipe,obj,kb cIn;
    class lex,eng cEn;
    class frames cFr;
```

By default `build` also runs the update, which is why the two stages read as one in ordinary use. They are separable, and worth separating whenever the grid needs thought: build without updating, inspect the analytic moments, then update on a grid you chose.
