# Tensor Component

This repository contains an `Tensor` component for my CSE 2231 portfolio project. The component models a mutable
2-dimensional tensor of `double` values stored conceptually as a rows-by-columns
matrix.

The design follows the software sequence discipline:

- `TensorKernel` defines the kernel operations and extends `Standard<Tensor>`.
- `Tensor` extends the kernel interface with secondary operations.
- `TensorSecondary` implements secondary and common methods strictly through the
  public kernel/Standard interface.
- `Tensor1` is the concrete kernel implementation backed by a flat row-major
  `double[]` representation.

## Features

The final component supports the following behavior:

- Standard operations: `newInstance`, `clear`, and `transferFrom`
- Kernel operations: `get`, `set`, `shape`, `isZero`, `reshape`, and `setShape`
- Secondary operations: `add`, `scale`, `sum`, `fill`, and `multiply`
- Common methods: `toString`, `equals`, and `hashCode`

`isZero` uses `TensorKernel.ZERO_TOLERANCE` to avoid treating tiny floating-point
roundoff as meaningful tensor content.

## Project Layout

```text
src/components/tensor/
    Tensor.java
    Tensor1.java
    TensorKernel.java
    TensorSecondary.java

src/components/tensor/examples/
    TensorNeuralNetworkDemo.java
    TensorSensorGridDemo.java

test/components/tensor/
    Tensor1Test.java
    TensorTest.java

doc/
    01-component-brainstorming/
    02-component-proof-of-concept/
    03-component-interfaces/
    04-component-abstract-class/
    05-component-kernel-implementation/
    06-component-finishing-touches/
```

## Example Use Cases

Two complete examples are included in `src/components/tensor/examples`:

1. `TensorNeuralNetworkDemo` uses tensor multiplication, addition, and scaling to
   simulate a tiny linear layer: `input * weights + bias`.
2. `TensorSensorGridDemo` uses tensors to represent a calibrated 2D temperature
   grid, compute an average, convert Celsius readings to Fahrenheit, and reshape
   the grid into a flat log record.

## Tests

The test suite is split according to the Part 6 requirement:

- `Tensor1Test` covers the concrete implementation's constructors, Standard
  methods, and kernel methods.
- `TensorTest` covers the secondary methods and common `Object` methods through
  the public `Tensor` interface.

To run the tests in VSCode, add the required course libraries to `lib/`:

```text
lib/components.jar
lib/junit-4.13.2.jar
lib/hamcrest-core-1.3.jar
```

Then enable assertions with `-ea`; the provided VSCode settings already include
that VM argument.

## Documentation

The `doc/` directory contains the completed portfolio writeups for Parts 1-6.
Part 6 includes the final reflection, changelog instructions, and submission
workflow notes.
