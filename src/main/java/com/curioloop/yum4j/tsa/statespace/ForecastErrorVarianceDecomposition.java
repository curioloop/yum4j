package com.curioloop.yum4j.tsa.statespace;

public final class ForecastErrorVarianceDecomposition {

    private final double[][][] shares;

    public ForecastErrorVarianceDecomposition(double[][][] shares) {
        if (shares == null || shares.length == 0) {
            throw new IllegalArgumentException("shares must not be empty");
        }
        this.shares = copy(shares);
    }

    public int steps() {
        return shares.length;
    }

    public int responseCount() {
        return shares[0].length;
    }

    public int impulseCount() {
        return shares[0].length == 0 ? 0 : shares[0][0].length;
    }

    public double share(int horizon, int response, int impulse) {
        if (horizon < 1 || horizon > shares.length) {
            throw new IllegalArgumentException("horizon must be in [1, steps]");
        }
        return shares[horizon - 1][response][impulse];
    }

    public double[][][] shares() {
        return copy(shares);
    }

    private static double[][][] copy(double[][][] source) {
        double[][][] copy = new double[source.length][][];
        for (int step = 0; step < source.length; step++) {
            copy[step] = new double[source[step].length][];
            for (int row = 0; row < source[step].length; row++) {
                copy[step][row] = source[step][row].clone();
            }
        }
        return copy;
    }
}