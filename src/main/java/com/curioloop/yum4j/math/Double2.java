package com.curioloop.yum4j.math;

public value record Double2(double _1, double _2) {

    public static Double2 bound(double lower, double upper) {
        if (Double.isNaN(lower) || Double.isNaN(upper)) {
            throw new IllegalArgumentException("bound must not be NaN");
        }
        if (lower > upper) {
            throw new IllegalArgumentException(
                    "bound require lower <= upper: lower=" + lower + ", upper=" + upper
            );
        }
        return new Double2(lower, upper);
    }

    public static Double2 fraction(double numerator, double denominator) {
        if (Double.isNaN(numerator) || Double.isNaN(denominator)) {
            throw new IllegalArgumentException("fraction must not be NaN");
        }
        return new Double2(numerator, denominator);
    }

    public double lower() {
        return _1;
    }

    public double upper() {
        return _2;
    }

    public double numerator() {
        return _1;
    }

    public double denominator() {
        return _2;
    }

}