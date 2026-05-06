package com.curioloop.yum4j.math;

/**
 * Shared low-level helpers used across special-function kernels.
 */
final class Commons {

    private static final double INTEGER_EPSILON = 1.0e-12;
    private static final int MAX_FACTORIAL = 170;
    private static final double[] FACTORIALS = createFactorialTable();

    private Commons() {
    }

    static boolean sameParameter(double left, double right) {
        return Math.abs(left - right) <= INTEGER_EPSILON * Math.max(1.0, Math.max(Math.abs(left), Math.abs(right)));
    }

    static boolean isInteger(double value) {
        return Math.abs(value - Math.rint(value)) <= INTEGER_EPSILON * Math.max(1.0, Math.abs(value));
    }

    static void validateFinite(double value, String label) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(label + " must be finite: " + value);
        }
    }

    static void validateProbability(double value, String label) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(label + " must be in [0, 1]: " + value);
        }
    }

    static void validateHolder(double[] holder, String label) {
        if (holder != null && holder.length == 0) {
            throw new IllegalArgumentException(label + " must have length >= 1 when provided");
        }
    }

    static void validateHolder(int[] holder, String label) {
        if (holder != null && holder.length == 0) {
            throw new IllegalArgumentException(label + " must have length >= 1 when provided");
        }
    }

    static double factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative: " + n);
        }
        if (n <= MAX_FACTORIAL) {
            return uncheckedFactorial(n);
        }

        double value = Gamma.tgamma(n + 1.0);
        if (!Double.isFinite(value)) {
            return value;
        }
        return Math.floor(value + 0.5);
    }

    static double uncheckedFactorial(int n) {
        return FACTORIALS[n];
    }

    static double sinPi(double x) {
        if (!Double.isFinite(x)) {
            return Double.NaN;
        }
        if (x < 0.0) {
            return -sinPiPositive(-x);
        }
        return sinPiPositive(x);
    }

    static double cosPi(double x) {
        if (!Double.isFinite(x)) {
            return Double.NaN;
        }

        boolean invert = false;
        if (Math.abs(x) < 0.25) {
            return Math.cos(Math.PI * x);
        }

        if (x < 0.0) {
            x = -x;
        }

        double whole = Math.floor(x);
        if (hasOddParity(whole)) {
            invert = !invert;
        }

        double remainder = x - whole;
        if (remainder > 0.5) {
            remainder = 1.0 - remainder;
            invert = !invert;
        }
        if (remainder == 0.5) {
            return 0.0;
        }

        double value;
        if (remainder > 0.25) {
            value = Math.sin(Math.PI * (0.5 - remainder));
        } else {
            value = Math.cos(Math.PI * remainder);
        }
        return invert ? -value : value;
    }

    private static double sinPiPositive(double x) {
        if (x < 0.5) {
            return Math.sin(Math.PI * x);
        }

        boolean invert;
        if (x < 1.0) {
            invert = true;
            x = -x;
        } else {
            invert = false;
        }

        double whole = Math.floor(x);
        if (hasOddParity(whole)) {
            invert = !invert;
        }

        double remainder = x - whole;
        if (remainder > 0.5) {
            remainder = 1.0 - remainder;
        }
        if (remainder == 0.5) {
            return invert ? -1.0 : 1.0;
        }

        double value = Math.sin(Math.PI * remainder);
        return invert ? -value : value;
    }

    private static boolean hasOddParity(double integralValue) {
        return Math.abs(Math.floor(integralValue * 0.5) * 2.0 - integralValue) > Math.ulp(1.0);
    }

    private static double[] createFactorialTable() {
        double[] values = new double[MAX_FACTORIAL + 1];
        values[0] = 1.0;
        for (int n = 1; n <= MAX_FACTORIAL; n++) {
            values[n] = values[n - 1] * n;
        }
        return values;
    }
}