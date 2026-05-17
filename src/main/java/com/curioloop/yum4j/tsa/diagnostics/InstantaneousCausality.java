package com.curioloop.yum4j.tsa.diagnostics;

public record InstantaneousCausality(int caused,
                                    int causing,
                                    double statistic,
                                    double pValue,
                                    int degreesOfFreedom) {
}