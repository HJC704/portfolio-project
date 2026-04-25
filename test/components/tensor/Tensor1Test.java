package components.tensor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * JUnit tests for the {@link Tensor1} kernel implementation. These tests cover
 * the Standard methods and kernel methods directly.
 */
public class Tensor1Test {

    /** Tolerance for comparing tensor entries in tests. */
    private static final double DELTA = 1.0E-12;

    /**
     * Creates a 2-by-3 tensor containing the values 1.0 through 6.0 in
     * row-major order.
     *
     * @return a non-zero 2-by-3 tensor
     */
    private static Tensor createTwoByThreeTensor() {
        Tensor tensor = new Tensor1(2, 3);
        double value = 1.0;
        int[] shape = tensor.shape();
        for (int row = 0; row < shape[0]; row++) {
            for (int col = 0; col < shape[1]; col++) {
                tensor.set(value, row, col);
                value++;
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
    public void testNoArgumentConstructorInitializesOneByOneZero() {
        Tensor tensor = new Tensor1();

        assertShape(tensor, 1, 1);
        assertEntry(0.0, tensor, 0, 0);
        assertTrue(tensor.isZero());
    }

    @Test
    public void testShapeConstructorInitializesRequestedZeroTensor() {
        Tensor tensor = new Tensor1(3, 2);

        assertShape(tensor, 3, 2);
        assertTrue(tensor.isZero());
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 2; col++) {
                assertEntry(0.0, tensor, row, col);
            }
        }
    }

    @Test
    public void testShapeReturnsDefensiveCopy() {
        Tensor tensor = new Tensor1(4, 5);
        int[] shape = tensor.shape();

        shape[0] = 99;
        shape[1] = 100;

        assertShape(tensor, 4, 5);
    }

    @Test
    public void testSetAndGetSingleEntryPreservesOtherEntries() {
        Tensor tensor = new Tensor1(2, 2);

        tensor.set(7.25, 1, 0);

        assertEntry(0.0, tensor, 0, 0);
        assertEntry(0.0, tensor, 0, 1);
        assertEntry(7.25, tensor, 1, 0);
        assertEntry(0.0, tensor, 1, 1);
    }

    @Test
    public void testIsZeroUsesToleranceForTinyRoundoff() {
        Tensor tensor = new Tensor1(1, 2);

        tensor.set(TensorKernel.ZERO_TOLERANCE / 2.0, 0, 0);
        assertTrue(tensor.isZero());

        tensor.set(TensorKernel.ZERO_TOLERANCE * 2.0, 0, 1);
        assertFalse(tensor.isZero());
    }

    @Test
    public void testIsZeroAcceptsValueExactlyAtTolerance() {
        Tensor tensor = new Tensor1(1, 1);

        tensor.set(TensorKernel.ZERO_TOLERANCE, 0, 0);

        assertTrue(tensor.isZero());
    }

    @Test
    public void testIsZeroAcceptsNegativeValueExactlyAtTolerance() {
        Tensor tensor = new Tensor1(1, 1);

        tensor.set(-TensorKernel.ZERO_TOLERANCE, 0, 0);

        assertTrue(tensor.isZero());
    }

    @Test
    public void testReshapePreservesRowMajorValues() {
        Tensor tensor = createTwoByThreeTensor();

        tensor.reshape(3, 2);

        assertShape(tensor, 3, 2);
        assertEntry(1.0, tensor, 0, 0);
        assertEntry(2.0, tensor, 0, 1);
        assertEntry(3.0, tensor, 1, 0);
        assertEntry(4.0, tensor, 1, 1);
        assertEntry(5.0, tensor, 2, 0);
        assertEntry(6.0, tensor, 2, 1);
    }

    @Test
    public void testSetShapeReplacesContentWithZeros() {
        Tensor tensor = createTwoByThreeTensor();

        tensor.setShape(4, 1);

        assertShape(tensor, 4, 1);
        assertTrue(tensor.isZero());
        for (int row = 0; row < 4; row++) {
            assertEntry(0.0, tensor, row, 0);
        }
    }

    @Test
    public void testNewInstanceProducesSameDynamicTypeAndInitialValue() {
        Tensor tensor = new Tensor1(2, 3);
        tensor.set(1.0, 0, 0);

        Tensor newTensor = tensor.newInstance();

        assertNotSame(tensor, newTensor);
        assertEquals(tensor.getClass(), newTensor.getClass());
        assertShape(newTensor, 1, 1);
        assertTrue(newTensor.isZero());
    }

    @Test
    public void testClearResetsToInitialValue() {
        Tensor tensor = createTwoByThreeTensor();

        tensor.clear();

        assertShape(tensor, 1, 1);
        assertEntry(0.0, tensor, 0, 0);
        assertTrue(tensor.isZero());
    }

    @Test
    public void testTransferFromMovesStateAndClearsSource() {
        Tensor destination = new Tensor1(1, 1);
        Tensor source = createTwoByThreeTensor();

        destination.transferFrom(source);

        assertShape(destination, 2, 3);
        assertEntry(1.0, destination, 0, 0);
        assertEntry(6.0, destination, 1, 2);
        assertShape(source, 1, 1);
        assertTrue(source.isZero());
    }

}
