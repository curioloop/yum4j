package com.curioloop.yum4j.tsa.sarimax;

import java.util.Arrays;

public final class SARIMAXPrediction {

    private final int start;
    private final int end;
    private final boolean dynamic;
    private final double[] mean;
    private final double[] variance;

    SARIMAXPrediction(int start, int end, boolean dynamic, double[] mean, double[] variance) {
        this.start = start;
        this.end = end;
        this.dynamic = dynamic;
        this.mean = mean;
        this.variance = variance;
    }

    public int start() {
        return start;
    }

    public int end() {
        return end;
    }

    public boolean dynamic() {
        return dynamic;
    }

    public double[] mean() {
        return Arrays.copyOf(mean, mean.length);
    }

    public double[] variance() {
        return Arrays.copyOf(variance, variance.length);
    }
}