package com.curioloop.yum4j.tsa.statespace;

import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;

import java.util.Arrays;
import java.util.Objects;

public final class SimulationSmootherRepetitions {

    private final SimulationSmootherResult[] results;

    public SimulationSmootherRepetitions(SimulationSmootherResult[] results) {
        Objects.requireNonNull(results, "results");
        if (results.length == 0) {
            throw new IllegalArgumentException("results must not be empty");
        }
        this.results = new SimulationSmootherResult[results.length];
        for (int i = 0; i < results.length; i++) {
            this.results[i] = Objects.requireNonNull(results[i], "result").copy();
        }
    }

    public int repetitions() {
        return results.length;
    }

    public int nsimulations() {
        return results[0].nobs;
    }

    public SimulationSmootherResult result(int repetition) {
        if (repetition < 0 || repetition >= results.length) {
            throw new IllegalArgumentException("repetition out of range: " + repetition);
        }
        return results[repetition].copy();
    }

    public SimulationSmootherResult[] results() {
        return Arrays.stream(results)
            .map(SimulationSmootherResult::copy)
            .toArray(SimulationSmootherResult[]::new);
    }
}