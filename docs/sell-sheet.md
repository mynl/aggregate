# `aggregate`

*`aggregate` is the electric drivetrain of loss modeling*. Simulation is the internal combustion engine of actuarial work. It runs on anything, goes anywhere, and burns runtime to get there: more simulations buy accuracy at the cost of speed, fewer buy speed at the cost of simulation noise. `aggregate` is the electric alternative. One drive mechanism, almost no moving parts, instant torque, and essentially exact output in milliseconds. Modeling faces a trilemma — speed, accuracy, flexibility: pick two. Simulation picks flexibility and trades speed and accuracy. We pick speed and accuracy, for flexibility, but we make no apologies: `aggregate` does one thing but does it astonishingly well.

## The engine: FFT convolution

Under the hood sits fast Fourier transform (FFT) convolution, a miraculously effective convolution algorithm. The FFT engine adds independent losses, compounds frequency and severity into univariate or bivariate aggregate loss distributions, prices gross, ceded, and net across a reinsurance tower, handles multi-line portfolios with shared frequency or correlated severity, and computes the conditional expectations (kappas) that drive distortion-based pricing and capital allocation. No sampling error, no convergence diagnostics, no overnight runs. 

## The controls: DecL

A model is a sentence, not a spreadsheet. The DecL domain-specific declarative language reads the way an actuary or underwriter thinks: 

```
agg Auto 8 claims 2000 xs 0 sev lognorm 50 cv 1.75 poisson
```

No maze of named arguments to memorize and no point-and-click. Model specifications are plain text: readable, diffable, auditable, and version-controlled; the input file is the model documentation. Need a thousand variants for a portfolio study? Script a thousand lines, not a thousand clicks.

## The instrumentation: validation

A serious machine tells you how it is running. Every `aggregate` object reports its mean, coefficient of variation, and skewness alongside the exact theoretical values, so the computed distribution is checked against a known standard on every build. When the discretization is pushed outside its envelope, the dashboard says so before you rely on the output. Confidence is built in, not bolted on.

## The drivetrain: exhibits and charting

The engine produces torque; the drivetrain puts it on the road. Raw output arrives as standard pandas DataFrames and NumPy arrays that any modern Python workflow can manipulate directly. On top of that raw material, the exhibits and charting modules build an intermediate representation of business-ready output, rendered by the front end of your choice: HTML, PDF via TeX, CSV, or a spreadsheet grid. Business logic is deliberately separated from raw ingredients. Exhibit design is like tire choice — nobody fits winter tires in the tropics — so version one ships a few proven sets and, more importantly, the abstraction that lets you mount your own.

## The showroom: Aggregate Loss Lab

Take a test drive in the browser — no installation, no Python, no setup. The Aggregate Loss Lab shows finished exhibits and plots straight from the engine: spectral and distortion pricing rules, evaluation of a quoted price, and the profit-and-loss walk across the tower. Ten minutes in the Lab shows you the car you could build.

## Learn more 

`pip install aggregate` · docs at aggregate.readthedocs.io · test drive at https://app.mynl.com · blog.mynl.com
 