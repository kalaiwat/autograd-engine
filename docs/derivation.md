# Derivations

Hand-derived partials for every primitive in the engine. Each section states the
forward expression, the local partials with respect to each operand, and the
resulting backward rule (the term accumulated into each operand's `grad` given
an incoming `out.grad`).

Conventions: `L` is the scalar being differentiated, `out` the node produced by
the primitive, and `dL/dout` the gradient flowing in from downstream.

---

## Chain rule and accumulation

_TODO: why gradients accumulate (`+=`) rather than overwrite, and what that means for a node reused in multiple places._

---

## `operator+`

_TODO: forward, ∂out/∂lhs, ∂out/∂rhs, backward rule._

## `operator-`

_TODO: forward, ∂out/∂lhs, ∂out/∂rhs, backward rule (note the sign on the rhs)._

## `operator*`

_TODO: forward, ∂out/∂lhs, ∂out/∂rhs, backward rule._

## `operator/`

_TODO: forward, ∂out/∂lhs, ∂out/∂rhs, backward rule, and behaviour as the denominator approaches zero._

## Unary negation

_TODO: forward, ∂out/∂operand, backward rule._

---

## `pow`

_TODO: forward, partial for a constant exponent, and what changes if the exponent is itself a `Value`._

## `exp`

_TODO: forward, ∂out/∂operand, backward rule expressed in terms of `out.data`._

## `log`

_TODO: forward, ∂out/∂operand, domain restriction._

---

## `tanh`

_TODO: forward, ∂out/∂operand, backward rule in terms of `out.data`._

## `relu`

_TODO: forward, partials on each side of the kink, and the convention chosen at exactly zero._

## `sigmoid`

_TODO: forward, ∂out/∂operand, backward rule in terms of `out.data`._

---

## Reductions

_TODO: `sum` and `mean` — how one incoming gradient is broadcast back over many operands._

## `matmul`

_TODO: forward, ∂L/∂A and ∂L/∂B in matrix form, and why the transposes fall out where they do._

---

## Numerical verification

_TODO: the finite-difference check used to confirm each rule above, and the tolerance it holds to._
