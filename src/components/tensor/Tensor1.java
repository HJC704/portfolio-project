package components.tensor;

/**
 * {@code Tensor} represented as a flat row-major {@code double[]} array
 * with implementations of primary methods.
 *
 * @convention
 *  $this.data != null and
 *  $this.rows > 0 and
 *  $this.cols > 0 and
 *  $this.data.length = $this.rows * $this.cols
 *
 * @correspondence
 *  this = [[$this.data[i * $this.cols + j] | 0 <= j < $this.cols]
 *                                           | 0 <= i < $this.rows]
 */
public class Tensor1 extends TensorSecondary {

    /*
     * Private members --------------------------------------------------------
     */

    /** Flat row-major backing array; element (i,j) = data[i * cols + j]. */
    private double[] data;

    /** Number of rows; always positive. */
    private int rows;

    /** Number of columns; always positive. */
    private int cols;

    /**
     * Creator of initial representation. Sets {@code this} to a 1x1 all-zero
     * tensor, consistent with the {@code @initially} clause of
     * {@link TensorKernel}.
     */
    private void createNewRep() {
        this.rows = 1;
        this.cols = 1;
        this.data = new double[1]; // Java zero-initializes double arrays
    }

    /*
     * Constructors -----------------------------------------------------------
     */

    /**
     * No-argument constructor. Produces a 1x1 all-zero tensor.
     */
    public Tensor1() {
        this.createNewRep();
    }

    /**
     * Constructs a {@code rows}-by-{@code cols} all-zero tensor.
     *
     * @param rows
     *            the number of rows
     * @param cols
     *            the number of columns
     */
    public Tensor1(int rows, int cols) {
        assert rows > 0 && cols > 0 : "Violation of: rows > 0 and cols > 0";
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    /*
     * Standard methods -------------------------------------------------------
     */

    @Override
    public final Tensor newInstance() {
        try {
            return this.getClass().getConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(
                    "Cannot construct object of type " + this.getClass());
        }
    }

    @Override
    public final void clear() {
        this.createNewRep();
    }

    @Override
    public final void transferFrom(Tensor source) {
        assert source != null : "Violation of: source is not null";
        assert source != this : "Violation of: source is not this";
        assert source instanceof Tensor1
                : "Violation of: source is of dynamic type Tensor1";
        Tensor1 localSource = (Tensor1) source;
        this.data = localSource.data;
        this.rows = localSource.rows;
        this.cols = localSource.cols;
        localSource.createNewRep();
    }

    /*
     * Kernel methods ---------------------------------------------------------
     */

    @Override
    public final double get(int row, int col) {
        assert 0 <= row && row < this.rows
                : "Violation of: 0 <= row < this.shape()[0]";
        assert 0 <= col && col < this.cols
                : "Violation of: 0 <= col < this.shape()[1]";
        return this.data[row * this.cols + col];
    }

    @Override
    public final void set(double value, int row, int col) {
        assert 0 <= row && row < this.rows
                : "Violation of: 0 <= row < this.shape()[0]";
        assert 0 <= col && col < this.cols
                : "Violation of: 0 <= col < this.shape()[1]";
        this.data[row * this.cols + col] = value;
    }

    @Override
    public final int[] shape() {
        return new int[] {this.rows, this.cols};
    }

    @Override
    public final boolean isZero() {
        boolean allZero = true;
        for (int k = 0; k < this.data.length && allZero; k++) {
            allZero = this.data[k] == 0.0;
        }
        return allZero;
    }

    @Override
    public final void reshape(int rows, int cols) {
        assert rows > 0 && cols > 0
                : "Violation of: rows > 0 and cols > 0";
        assert rows * cols == this.rows * this.cols
                : "Violation of: rows * cols = #this.shape()[0] * #this.shape()[1]";
        this.rows = rows;
        this.cols = cols;
        // data array is unchanged: row-major flat indices are preserved
    }

    @Override
    public final void setShape(int rows, int cols) {
        assert rows > 0 && cols > 0
                : "Violation of: rows > 0 and cols > 0";
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols]; // fresh array, all zeros
    }

}
