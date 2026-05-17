package com.curioloop.yum4j.tsa.varmax;

public value record VARMAXOrder(int autoregressive, int movingAverage) {

    public VARMAXOrder {
        if (autoregressive < 0) {
            throw new IllegalArgumentException("autoregressive order must be non-negative");
        }
        if (movingAverage < 0) {
            throw new IllegalArgumentException("movingAverage order must be non-negative");
        }
        if (autoregressive == 0 && movingAverage == 0) {
            throw new IllegalArgumentException("at least one of autoregressive or movingAverage must be positive");
        }
    }

    public static VARMAXOrder of(int autoregressive, int movingAverage) {
        return new VARMAXOrder(autoregressive, movingAverage);
    }
}
