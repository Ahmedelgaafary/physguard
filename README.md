# physguard

Verify LLM numerical answers against **units, physical laws and sanity bounds**, and feed failures
back to the model for repair.

LLMs often return numbers that look right but are wrong by a unit prefix (mm vs m), have the wrong
dimension, or violate basic physics. `physguard` turns an answer into a structured claim and checks it
deterministically, so wrong numbers are rejected before they reach a user or a downstream system.

## How it works

<img width="1039" height="659" alt="Screenshot (205)" src="https://github.com/user-attachments/assets/21b3b841-c0c9-4fc8-a590-5ce0aea6ff87" />


Checks, in order: parse -> universal limits (speed of light, absolute zero, negative mass) ->
user bounds -> expected dimension -> dimensional match with the law -> magnitude vs. the reference
(configurable tolerance) with a hint when the error is a power of ten.

Verdicts: `PASS` (verified against a law), `WARN` (no problem found but nothing to verify against),
`FAIL`. `WARN` is never reported as `PASS`.

## Quick start

```bash
pip install -e ".[dev]"
make test      # 40 tests
make demo      # repair loop with a scripted fake LLM (no API key)
make bench     # synthetic benchmark
```

```python
from physguard import Claim, Guard

claim = Claim(
    answer="1.6667 m",
    law="cantilever_tip_deflection",
    inputs={"F": "1 kN", "L": "2 m", "E": "200 GPa", "I": "8e-6 m**4"},
)
report = Guard(rtol=0.01).verify(claim)
print(report.verdict)      # Verdict.FAIL
print(report.feedback())   # ... answer is off by a factor of 10^3; likely a unit-prefix error
```

Plug in any LLM by passing an object with `complete(prompt: str) -> str` to `guarded_answer`.

## Add a law

```python
from physguard.laws import law

@law("spring_energy", ("k", "x"))
def spring_energy(k, x):
    """Elastic energy: 0.5 k x^2."""
    return (0.5 * k * x**2).to("J")
```

## Benchmark (synthetic)

`benchmarks/run_benchmark.py` generates candidate answers with known error types over 10 scenarios.
Result: 50/50 wrong answers rejected, 0/20 correct answers rejected.

**Read this carefully:** the candidates are generated from the same reference laws the guard uses, so
this measures the checker's correctness, not real LLM behaviour. See "Next steps" for the real evaluation.

## Limitations

- Verifies only quantities covered by a registered law; everything else is at most `WARN`.
- The LLM must name the right law and inputs. A wrong law choice with consistent inputs can pass.
  A planned mitigation is a second check with an independent law or a numeric solver.
- Closed-form laws only; no numerical solvers yet.

## Next steps (roadmap)

- [ ] Real evaluation: 100+ engineering questions answered by 2-3 LLMs, guard on vs. off, report
      error-acceptance rate and repair success rate
- [ ] Solver-backed checks (SciPy ODE/FEM) and conservation-law checks (energy, momentum)
- [ ] Arabic unit phrases ("5 كيلو نيوتن")
- [ ] Publish to PyPI, add docs site
- [ ] LangGraph verifier node (used by Project 2: engineering simulation copilot)

## Design decisions

- **pint** for unit algebra instead of hand-written conversions: fewer bugs, offset units handled.
- **Pydantic** claims: the LLM contract is explicit and validated.
- **No vendor SDK** in the core: the LLM client is a one-method protocol, easy to test and swap.
- **Deterministic verdicts**: no LLM-as-judge in the loop that checks numbers.

## Law packs (v0.2)

21 laws: beams (2), Ohm's law, kinetic energy, projectile range, ideal gas, hydrostatic pressure, plus
nuclear (decay constant, activity, gamma attenuation, inverse-square dose, four-factor formula,
reactivity with `pcm` unit, reactor period, Way-Wigner decay heat, fuel centerline temperature),
conduction through a wall, Darcy-Weisbach pressure drop, thin-wall hoop stress, three-phase power and
Euler buckling. See `src/physguard/laws_extra.py`. 40 tests.
