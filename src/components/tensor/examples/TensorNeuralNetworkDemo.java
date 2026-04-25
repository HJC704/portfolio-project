package components.tensor.examples;

import components.tensor.Tensor;
import components.tensor.Tensor1;

/**
 * Demonstrates the Tensor component as the core data structure for a tiny
 * neural-network-style linear layer.
 */
public final class TensorNeuralNetworkDemo {

    /** Private constructor to prevent instantiation. */
    private TensorNeuralNetworkDemo() {
        // Utility class.
    }

    /**
     * Populates a tensor row using the given values.
     *
     * @param tensor
     *            the tensor to update
     * @param row
     *            the row to populate
     * @param values
     *            the row values
     */
    private static void setRow(Tensor tensor, int row, double[] values) {
        for (int col = 0; col < values.length; col++) {
            tensor.set(values[col], row, col);
        }
    }

    /**
     * Runs the demonstration.
     *
     * @param args
     *            command-line arguments; unused
     */
    public static void main(String[] args) {
        Tensor input = new Tensor1(1, 3);
        setRow(input, 0, new double[] {0.8, 0.1, 0.6});

        Tensor weights = new Tensor1(3, 2);
        setRow(weights, 0, new double[] {0.20, -0.40});
        setRow(weights, 1, new double[] {0.70, 0.10});
        setRow(weights, 2, new double[] {-0.30, 0.90});

        Tensor bias = new Tensor1(1, 2);
        setRow(bias, 0, new double[] {0.05, -0.10});

        Tensor logits = input.multiply(weights);
        logits.add(bias);
        logits.scale(0.5);

        System.out.println("Input feature vector:");
        System.out.print(input);
        System.out.println("Linear layer output after bias and scaling:");
        System.out.print(logits);
    }

}
