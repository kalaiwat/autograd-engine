# autograd-engine

A reverse-mode automatic differentiation engine in C++17, written from scratch with
no dependencies outside the standard library (Catch2 is used by the tests only). The
unit of computation is a scalar: a `Value` holds one `double` of `data` and one
`double` of `grad`. Expressions are built by overloaded operators on
`std::shared_ptr<Value>`, and each result node records the operands it was built from
alongside a closure implementing its own local backward rule. Calling `backward()` on
a node walks its operands depth-first to produce a topological order, zeroes the
gradient of every node it reaches, seeds the root with `grad = 1.0`, then replays that
order in reverse, each closure accumulating into its operands with `+=` — so a node
used more than once sums the contributions from every path it appears on.

## Build and test

_TODO: the CMake configure/build/ctest invocations, and what the Debug (sanitizer) and Release configurations each cover._

## Verification results

_TODO: finite-difference gradient-check results — primitives covered, step size, tolerance, worst-case relative error._

## Derivations

_TODO: link to [docs/derivation.md](docs/derivation.md) and note which primitives are derived there._

## Ownership and memory design

The scalar stage uses `shared_ptr` reference counting rather than an arena with index
handles. Nodes are produced one at a time by operator overloads that must return
something the caller owns, lifetimes are non-uniform (leaves generally outlive the
expressions built from them), and reference counting expresses that directly at the
point of construction. An arena would buy locality and O(1) teardown, and is the
natural change to make once nodes become tensors allocated in bulk — but it forces
every node to be addressed by index and requires the arena to outlive every handle
into it, which is a lot of machinery for graphs of a few dozen scalars.

Ownership runs one way. A result node owns its operands, through
`std::vector<std::shared_ptr<Value>>`; operands never point back up at the result they
feed. Holding the output of an expression therefore keeps everything beneath it alive,
and releasing that output releases the whole subgraph apart from any leaves the caller
still holds. Because the owning edges only ever run from result to operand, they form
a DAG. Reusing one node twice in an expression (`x * x`) simply stores it twice and
its reference count reflects that.

The one place a cycle could form is the backward closure, which is why it captures a
raw pointer:

```cpp
result_node->backward_fn = [self = result_node.get()]() {
    self->operands[0]->grad += (self->grad * self->operands[1]->data);
    self->operands[1]->grad += (self->grad * self->operands[0]->data);
};
```

Each node stores its own `backward_fn`, and that closure has to read and write the node
it lives in — `self->grad`, `self->operands`. Capturing `result_node` itself would put
a `shared_ptr` to the node inside a `std::function` held by that same node: a one-node
cycle whose reference count never reaches zero. This is exactly the case reference
counting cannot detect on its own — there is no cycle collector to notice it later, so
the node and everything it transitively owns would leak silently, and the leak would be
invisible in the program's behaviour. Capturing `result_node.get()` makes the
back-reference non-owning, so the cycle never exists in the first place.

The raw `self` is safe because it cannot outlive what it points at. It lives inside the
node's own `backward_fn`, so it is destroyed with the node, and it is only dereferenced
during `Value::backward()`, which reaches a node only by walking owning edges down from
a root the caller is holding. The operands that closure touches are reached through
`self->operands`, which are owning pointers, so they are guaranteed alive for the
duration of the call. A `weak_ptr` capture would break the cycle too, but it would pay a
`lock()` on every edge of every backward pass to re-establish a lifetime guarantee the
graph's ownership already provides.

## Graph visualization

_TODO: rendered computation graph for a small expression, plus the command that produces it._

## Loss curve

_TODO: training loss over epochs for the demo model, with the setup that produced it._

## Performance

_TODO: matmul benchmark table — size, time, GFLOPS, and the machine it was measured on._
