package components.tensor.examples;

import components.tensor.Tensor;
import components.tensor.Tensor1;

/**
 * Demonstrates the Tensor component as a 2D sensor grid for simple temperature
 * calibration, aggregation, and logging.
 */
public final class TensorSensorGridDemo {

    /** Private constructor to prevent instantiation. */
    private TensorSensorGridDemo() {
        // Utility class.
    }

    /**
     * Creates a copy of a tensor through the public component API.
     *
     * @param source
     *            the tensor to copy
     * @return a tensor with the same shape and values as {@code source}
     */
    private static Tensor copyOf(Tensor source) {
        int[] shape = source.shape();
        Tensor result = new Tensor1(shape[0], shape[1]);
        for (int row = 0; row < shape[0]; row++) {
            for (int col = 0; col < shape[1]; col++) {
                result.set(source.get(row, col), row, col);
            }
        }
        return result;
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
        Tensor celsius = new Tensor1(3, 3);
        setRow(celsius, 0, new double[] {21.3, 21.7, 22.0});
        setRow(celsius, 1, new double[] {20.9, 21.5, 22.4});
        setRow(celsius, 2, new double[] {20.5, 21.1, 21.8});

        Tensor calibrationOffset = new Tensor1(3, 3);
        calibrationOffset.fill(0.2);
        calibrationOffset.set(-0.1, 1, 2);

        celsius.add(calibrationOffset);
        double averageCelsius = celsius.sum() / 9.0;

        Tensor fahrenheit = copyOf(celsius);
        fahrenheit.scale(1.8);
        Tensor freezingOffset = new Tensor1(3, 3);
        freezingOffset.fill(32.0);
        fahrenheit.add(freezingOffset);

        Tensor flatLogRecord = copyOf(fahrenheit);
        flatLogRecord.reshape(1, 9);

        System.out.println("Calibrated Celsius grid:");
        System.out.print(celsius);
        System.out.printf("Average Celsius: %.2f%n%n", averageCelsius);
        System.out.println("Fahrenheit grid:");
        System.out.print(fahrenheit);
        System.out.println("Flattened record for logging:");
        System.out.print(flatLogRecord);
    }

}
