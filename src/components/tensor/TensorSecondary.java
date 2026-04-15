package components.tensor;

/**
 * Layered implementations of secondary methods for {@code Tensor}.
 *
 * <p>
 * Every method here is implemented using only the kernel methods declared in
 * {@link TensorKernel} ({@code get}, {@code set}, {@code shape},
 * {@code isZero}, {@code reshape}, {@code setShape}) and the
 * {@code Standard} methods inherited through the hierarchy
 * ({@code clear}, {@code newInstance}, {@code transferFrom}).
 * No method in this class accesses the underlying representation directly.
 * </p>
 */
public abstract class TensorSecondary implements Tensor {

    /*
     * Common methods
     */

    @Override
    public String toString() {
        int[] s = this.shape();
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor[").append(s[0]).append("x").append(s[1])
                .append("]:\n");
        for (int i = 0; i < s[0]; i++) {
            sb.append("  [");
            for (int j = 0; j < s[1]; j++) {
                sb.append(String.format("%7.3f", this.get(i, j)));
                if (j < s[1] - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]\n");
        }
        return sb.toString();
    }


    @Override
    public boolean equals(Object obj) {
        boolean eq = obj instanceof Tensor;
        if (eq) {
            Tensor other = (Tensor) obj;
            int[] s1 = this.shape();
            int[] s2 = other.shape();
            eq = s1[0] == s2[0] && s1[1] == s2[1];
            for (int i = 0; i < s1[0] && eq; i++) {
                for (int j = 0; j < s1[1] && eq; j++) {
                    eq = Double.compare(this.get(i, j), other.get(i, j)) == 0;
                }
            }
        }
        return eq;
    }


    @Override
    public int hashCode() {
        int[] s = this.shape();
        int result = 31 * s[0] + s[1];
        for (int i = 0; i < s[0]; i++) {
            for (int j = 0; j < s[1]; j++) {
                result = 31 * result + Double.hashCode(this.get(i, j));
            }
        }
        return result;
    }

    /*
     * Other non-kernel methods
     */


    @Override
    public void add(Tensor t) {
        assert t != null : "Violation of: t is not null";
        int[] s = this.shape();
        assert t.shape()[0] == s[0] && t.shape()[1] == s[1]
                : "Violation of: t.shape()[0] = this.shape()[0]"
                        + " and t.shape()[1] = this.shape()[1]";
        for (int i = 0; i < s[0]; i++) {
            for (int j = 0; j < s[1]; j++) {
                this.set(this.get(i, j) + t.get(i, j), i, j);
            }
        }
    }


    @Override
    public void scale(double scalar) {
        int[] s = this.shape();
        for (int i = 0; i < s[0]; i++) {
            for (int j = 0; j < s[1]; j++) {
                this.set(this.get(i, j) * scalar, i, j);
            }
        }
    }


    @Override
    public double sum() {
        int[] s = this.shape();
        double total = 0.0;
        for (int i = 0; i < s[0]; i++) {
            for (int j = 0; j < s[1]; j++) {
                total += this.get(i, j);
            }
        }
        return total;
    }


    @Override
    public void fill(double value) {
        int[] s = this.shape();
        for (int i = 0; i < s[0]; i++) {
            for (int j = 0; j < s[1]; j++) {
                this.set(value, i, j);
            }
        }
    }


    @Override
    public Tensor multiply(Tensor b) {
        assert b != null : "Violation of: b is not null";
        int[] s1 = this.shape();
        int[] s2 = b.shape();
        assert s1[1] == s2[0]
                : "Violation of: this.shape()[1] = b.shape()[0]";
        Tensor result = this.newInstance();
        result.setShape(s1[0], s2[1]);
        for (int i = 0; i < s1[0]; i++) {
            for (int j = 0; j < s2[1]; j++) {
                double val = 0.0;
                for (int k = 0; k < s1[1]; k++) {
                    val += this.get(i, k) * b.get(k, j);
                }
                result.set(val, i, j);
            }
        }
        return result;
    }

}
