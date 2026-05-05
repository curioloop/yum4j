package com.curioloop.yum4j.tsa.sarimax;

/**
 * Seasonal ARIMA order $(P, D, Q, s)$.
 */
public value record SeasonalOrder(int autoregressive, int integration, int movingAverage, int period) {

    public SeasonalOrder {
        if (autoregressive < 0) {
            throw new IllegalArgumentException("seasonal autoregressive order must be non-negative");
        }
        if (integration < 0) {
            throw new IllegalArgumentException("seasonal integration order must be non-negative");
        }
        if (movingAverage < 0) {
            throw new IllegalArgumentException("seasonal movingAverage order must be non-negative");
        }
        if ((autoregressive > 0 || integration > 0 || movingAverage > 0) && period <= 1) {
            throw new IllegalArgumentException("seasonal period must be greater than 1 when seasonal terms are present");
        }
        if (autoregressive == 0 && integration == 0 && movingAverage == 0 && period < 0) {
            throw new IllegalArgumentException("seasonal period must be non-negative");
        }
    }

    public static SeasonalOrder of(int autoregressive, int integration, int movingAverage, int period) {
        return new SeasonalOrder(autoregressive, integration, movingAverage, period);
    }

    public static SeasonalOrder none() {
        return new SeasonalOrder(0, 0, 0, 0);
    }

    public boolean isZero() {
        return autoregressive == 0 && integration == 0 && movingAverage == 0;
    }
}