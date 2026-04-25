# Source Folder

This folder contains the Tensor component source code.

```text
src/components/tensor/
    TensorKernel.java      kernel interface
    Tensor.java            enhanced interface
    TensorSecondary.java   layered secondary implementation
    Tensor1.java           row-major array kernel implementation

src/components/tensor/examples/
    TensorNeuralNetworkDemo.java
    TensorSensorGridDemo.java
```

The component follows the OSU Components discipline: clients program through the
interfaces, secondary operations are layered over kernel and Standard methods,
and the concrete representation is isolated inside `Tensor1`.
