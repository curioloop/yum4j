package com.curioloop.yum4j.math;

public value record DoubleI(double _1, int _2) {

    public static DoubleI signed(double value, int sign) {
        if (sign != -1 && sign != 1) {
            throw new IllegalArgumentException("sign must be either -1 or 1: " + sign);
        }
        return new DoubleI(value, sign);
    }

    public static DoubleI flagged(double value, boolean flag) {
        return new DoubleI(value, flag ? 1 : 0);
    }

    public double value() {
        return _1;
    }

    public int sign() {
        return _2;
    }

    public boolean flag() {
        return _2 != 0;
    }

}
