package com.curioloop.yum4j.tsa.statespace;

public final class ImpulseResponse {

    private final double[][][] responses;
    private final boolean orthogonalized;
    private final boolean cumulative;

    public ImpulseResponse(double[][][] responses, boolean orthogonalized, boolean cumulative) {
        if (responses == null || responses.length == 0) {
            throw new IllegalArgumentException("responses must not be empty");
        }
        this.responses = copy(responses);
        this.orthogonalized = orthogonalized;
        this.cumulative = cumulative;
    }

    public int steps() {
        return responses.length - 1;
    }

    public int responseCount() {
        return responses[0].length;
    }

    public int impulseCount() {
        return responses[0].length == 0 ? 0 : responses[0][0].length;
    }

    public boolean orthogonalized() {
        return orthogonalized;
    }

    public boolean cumulative() {
        return cumulative;
    }

    public double[][][] responses() {
        return copy(responses);
    }

    public double[][] response(int step) {
        if (step < 0 || step >= responses.length) {
            throw new IllegalArgumentException("step out of range: " + step);
        }
        double[][] source = responses[step];
        double[][] copy = new double[source.length][];
        for (int row = 0; row < source.length; row++) {
            copy[row] = source[row].clone();
        }
        return copy;
    }

    private static double[][][] copy(double[][][] source) {
        double[][][] copy = new double[source.length][][];
        for (int step = 0; step < source.length; step++) {
            if (source[step] == null) {
                throw new IllegalArgumentException("response steps must not be null");
            }
            copy[step] = new double[source[step].length][];
            for (int row = 0; row < source[step].length; row++) {
                if (source[step][row] == null) {
                    throw new IllegalArgumentException("response rows must not be null");
                }
                copy[step][row] = source[step][row].clone();
            }
        }
        return copy;
    }
}