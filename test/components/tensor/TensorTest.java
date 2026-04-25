package components.tensor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * JUnit tests for the secondary and common methods of {@link Tensor}. These
 * tests exercise methods implemented in {@link TensorSecondary} through the
 * concrete {@link Tensor1} implementation.
 */
public class TensorTest {

    /** Tolerance for comparing tensor entries in tests. */
    private static final double DELTA = 1.0E-12;

    /**
     * Creates a tensor from a two-dimensional array of values.
     *
     * @param values
     *            the values to copy into the tensor
     * @return a tensor containing {@code values}
     */
    private static Tensor tensorFrom(double[][] values) {
        Tensor tensor = new Tensor1(values.length, values[0].length);
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                tensor.set(values[row][col], row, col);
            }
        }
        return tensor;
    }

    /**
     * Asserts that the given tensor has the requested shape.
     *
     * @param tensor
     *            the tensor to inspect
     * @param rows
     *            the expected row count
     * @param cols
     *            the expected column count
     */
    private static void assertShape(Tensor tensor, int rows, int cols) {
        assertArrayEquals(new int[] {rows, cols}, tensor.shape());
    }

    /**
     * Asserts that a tensor entry is equal to the expected value.
     *
     * @param expected
     *            the expected entry value
     * @param tensor
     *            the tensor to inspect
     * @param row
     *            the row index
     * @param col
     *            the column index
     */
    private static void assertEntry(double expected, Tensor tensor, int row,
            int col) {
        assertEquals(expected, tensor.get(row, col), DELTA);
    }

    @Test
    public void testAddUpdatesThisOnly() {
        Tensor left = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor right = tensorFrom(new double[][] {{5.0, 6.0}, {7.0, 8.0}});

        left.add(right);

        assertEntry(6.0, left, 0, 0);
        assertEntry(8.0, left, 0, 1);
        assertEntry(10.0, left, 1, 0);
        assertEntry(12.0, left, 1, 1);
        assertEntry(5.0, right, 0, 0);
        assertEntry(8.0, right, 1, 1);
    }

    @Test
    public void testScaleMultipliesEveryEntry() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, -2.0}, {3.5, 0.0}});

        tensor.scale(-2.0);

        assertEntry(-2.0, tensor, 0, 0);
        assertEntry(4.0, tensor, 0, 1);
        assertEntry(-7.0, tensor, 1, 0);
        assertEntry(-0.0, tensor, 1, 1);
    }

    @Test
    public void testSumReturnsTotalAndPreservesTensor() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, 2.5}, {-3.0, 4.0}});
        Tensor copy = tensorFrom(new double[][] {{1.0, 2.5}, {-3.0, 4.0}});

        double total = tensor.sum();

        assertEquals(4.5, total, DELTA);
        assertEquals(copy, tensor);
    }

    @Test
    public void testFillSetsEveryEntry() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, 2.0, 3.0}});

        tensor.fill(9.5);

        assertShape(tensor, 1, 3);
        assertEntry(9.5, tensor, 0, 0);
        assertEntry(9.5, tensor, 0, 1);
        assertEntry(9.5, tensor, 0, 2);
    }

    @Test
    public void testMultiplySquareMatrices() {
        Tensor left = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor right = tensorFrom(new double[][] {{5.0, 6.0}, {7.0, 8.0}});

        Tensor product = left.multiply(right);

        assertShape(product, 2, 2);
        assertEntry(19.0, product, 0, 0);
        assertEntry(22.0, product, 0, 1);
        assertEntry(43.0, product, 1, 0);
        assertEntry(50.0, product, 1, 1);
    }

    @Test
    public void testMultiplyRectangularMatricesPreservesOperands() {
        Tensor left = tensorFrom(new double[][] {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}});
        Tensor right = tensorFrom(new double[][] {{7.0, 8.0}, {9.0, 10.0},
                {11.0, 12.0}});
        Tensor leftCopy = tensorFrom(new double[][] {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}});
        Tensor rightCopy = tensorFrom(new double[][] {{7.0, 8.0}, {9.0, 10.0},
                {11.0, 12.0}});

        Tensor product = left.multiply(right);

        assertShape(product, 2, 2);
        assertEntry(58.0, product, 0, 0);
        assertEntry(64.0, product, 0, 1);
        assertEntry(139.0, product, 1, 0);
        assertEntry(154.0, product, 1, 1);
        assertEquals(leftCopy, left);
        assertEquals(rightCopy, right);
    }

    @Test
    public void testMultiplyRowVectorByMatrixProducesRowVector() {
        Tensor row = tensorFrom(new double[][] {{1.0, 2.0, 3.0}});
        Tensor matrix = tensorFrom(new double[][] {{1.0, 4.0}, {2.0, 5.0},
                {3.0, 6.0}});

        Tensor product = row.multiply(matrix);

        assertShape(product, 1, 2);
        assertEntry(14.0, product, 0, 0);
        assertEntry(32.0, product, 0, 1);
    }

    @Test
    public void testMultiplyMatrixByColumnVectorProducesColumnVector() {
        Tensor matrix = tensorFrom(new double[][] {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}});
        Tensor column = tensorFrom(new double[][] {{7.0}, {8.0}, {9.0}});

        Tensor product = matrix.multiply(column);

        assertShape(product, 2, 1);
        assertEntry(50.0, product, 0, 0);
        assertEntry(122.0, product, 1, 0);
    }

    @Test
    public void testToStringIncludesShapeAndFormattedValues() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, -2.5}});

        String text = tensor.toString();

        assertTrue(text.contains("Tensor[1x2]"));
        assertTrue(text.contains("  1.000"));
        assertTrue(text.contains(" -2.500"));
    }

    @Test
    public void testEqualsTrueForSameShapeAndValues() {
        Tensor first = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor second = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});

        assertTrue(first.equals(second));
        assertTrue(second.equals(first));
    }

    @Test
    public void testEqualsReflexive() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});

        assertTrue(tensor.equals(tensor));
    }

    @Test
    public void testEqualsFalseForNull() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, 2.0}});

        assertFalse(tensor.equals(null));
    }

    @Test
    public void testEqualsFalseForDifferentShape() {
        Tensor first = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor second = tensorFrom(new double[][] {{1.0, 2.0, 3.0, 4.0}});

        assertFalse(first.equals(second));
    }

    @Test
    public void testEqualsFalseForDifferentValues() {
        Tensor first = tensorFrom(new double[][] {{1.0, 2.0}});
        Tensor second = tensorFrom(new double[][] {{1.0, 3.0}});

        assertFalse(first.equals(second));
        assertFalse(first.equals("not a tensor"));
    }

    @Test
    public void testHashCodeMatchesForEqualTensors() {
        Tensor first = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor second = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});

        assertEquals(first.hashCode(), second.hashCode());
    }

    @Test
    public void testHashCodeStableAcrossCallsAndDoesNotMutate() {
        Tensor tensor = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        Tensor copy = tensorFrom(new double[][] {{1.0, 2.0}, {3.0, 4.0}});

        int firstCall = tensor.hashCode();
        int secondCall = tensor.hashCode();

        assertEquals(firstCall, secondCall);
        assertEquals(copy, tensor);
    }

}
