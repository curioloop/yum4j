package com.curioloop.yum4j.tsa.statespace;

import java.util.Arrays;
import java.util.Objects;

public final class ImpulseResponseRepetitions {

    private final ImpulseResponse[] responses;

    public ImpulseResponseRepetitions(ImpulseResponse[] responses) {
        Objects.requireNonNull(responses, "responses");
        if (responses.length == 0) {
            throw new IllegalArgumentException("responses must not be empty");
        }
        this.responses = responses.clone();
        for (ImpulseResponse response : this.responses) {
            Objects.requireNonNull(response, "response");
        }
    }

    public int repetitions() {
        return responses.length;
    }

    public int steps() {
        return responses[0].steps();
    }

    public int responseCount() {
        return responses[0].responseCount();
    }

    public int impulseCount() {
        return responses[0].impulseCount();
    }

    public ImpulseResponse response(int repetition) {
        if (repetition < 0 || repetition >= responses.length) {
            throw new IllegalArgumentException("repetition out of range: " + repetition);
        }
        return responses[repetition];
    }

    public ImpulseResponse[] responses() {
        return responses.clone();
    }

    public double[][][] mean() {
        double[][][] mean = responses[0].responses();
        for (int repetition = 1; repetition < responses.length; repetition++) {
            double[][][] values = responses[repetition].responses();
            for (int step = 0; step < mean.length; step++) {
                for (int row = 0; row < mean[step].length; row++) {
                    for (int col = 0; col < mean[step][row].length; col++) {
                        mean[step][row][col] += values[step][row][col];
                    }
                }
            }
        }
        for (double[][] step : mean) {
            for (double[] row : step) {
                for (int col = 0; col < row.length; col++) {
                    row[col] /= responses.length;
                }
            }
        }
        return mean;
    }

    @Override
    public String toString() {
        return "ImpulseResponseRepetitions" + Arrays.toString(responses);
    }
}