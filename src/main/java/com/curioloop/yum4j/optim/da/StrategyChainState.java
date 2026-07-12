/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import java.util.Random;

/**
 * Mutable strategy-chain state for dual annealing's local-search decisions.
 *
 * <p>The annealing core keeps the Markov-chain current point in {@link DAWorkspace};
 * this helper tracks the extra SciPy state used to decide when local search should run. It
 * remembers a chain-local minimum during long stagnation and controls the probabilistic gate
 * for launching a local minimizer from that point.</p>
 */
final class StrategyChainState {

    /** Number of decision variables. */
    int dimension;

    /** Best chain-local location used when long-stagnation local search is triggered. */
    double[] minimumLocation;

    /** Objective value at {@link #minimumLocation}. */
    double minimumEnergy;

    /** Temperature scaled for the current strategy-chain step; used by acceptance/local-search tests. */
    double temperatureStep;

    /** Number of strategy-chain runs since the last global/local-search improvement. */
    int notImproved;

    /** Threshold for forcing a local search on the chain minimum. */
    int notImprovedMax;

    /** SciPy-style K factor used by the probabilistic chain-minimum local-search gate. */
    int localSearchScale;

    /** True when the current strategy-chain run improved the global best. */
    boolean energyStateImproved;

    /** Ensures storage for a strategy chain that keeps chain-local minima. */
    void ensure(int dimension) {
        ensure(dimension, true);
    }

    /** Ensures storage, optionally disabling minimum tracking when local search is off. */
    void ensure(int dimension, boolean keepMinimum) {
        if (dimension <= 0) throw new IllegalArgumentException("dimension must be positive");
        if (!keepMinimum) {
            this.dimension = dimension;
            this.minimumLocation = null;
            return;
        }
        if (this.dimension == dimension && minimumLocation != null) return;
        this.dimension = dimension;
        this.minimumLocation = new double[dimension];
    }

    /** Resets counters and records the current location as the chain minimum. */
    void reset(double[] currentLocation, double currentEnergy, boolean keepMinimum) {
        ensure(currentLocation.length, keepMinimum);
        this.minimumEnergy = currentEnergy;
        if (minimumLocation != null) {
            System.arraycopy(currentLocation, 0, minimumLocation, 0, dimension);
        }
        this.temperatureStep = 0.0;
        this.notImproved = 0;
        this.notImprovedMax = 1000;
        this.localSearchScale = 100 * dimension;
        this.energyStateImproved = false;
    }

    /** Starts one outer annealing run at a new temperature. */
    void startRun(int step, double temperature) {
        temperatureStep = temperature / (step + 1.0);
        notImproved++;
        energyStateImproved = step == 0;
    }

    /** Records an accepted point as the current chain minimum candidate. */
    void accepted(double[] currentLocation, double currentEnergy) {
        if (minimumLocation == null) return;
        minimumEnergy = currentEnergy;
        System.arraycopy(currentLocation, 0, minimumLocation, 0, dimension);
    }

    /** Marks that this run improved the global best and resets stagnation. */
    void bestImproved() {
        energyStateImproved = true;
        notImproved = 0;
    }

    /** Updates the stored chain minimum while the chain is in long stagnation mode. */
    void maybeRecordStagnationMinimum(int innerStep, double[] currentLocation, double currentEnergy) {
        if (minimumLocation == null) return;
        if (notImproved >= notImprovedMax && (innerStep == 0 || currentEnergy < minimumEnergy)) {
            minimumEnergy = currentEnergy;
            System.arraycopy(currentLocation, 0, minimumLocation, 0, dimension);
        }
    }

    /** Decides whether to launch local search from the stored chain minimum. */
    boolean shouldRunChainLocalSearch(double bestEnergy, double currentEnergy, Random rng) {
        if (minimumLocation == null) return false;
        if (localSearchScale < 90 * dimension) {
            double probability = Math.exp(localSearchScale * (bestEnergy - currentEnergy) / temperatureStep);
            if (probability >= rng.nextDouble()) return true;
        }
        return notImproved >= notImprovedMax;
    }

    /** Records a successful local search and tightens the stagnation threshold. */
    void localSearchImproved(double[] location, double energy) {
        if (minimumLocation == null) return;
        minimumEnergy = energy;
        System.arraycopy(location, 0, minimumLocation, 0, dimension);
        notImproved = 0;
        notImprovedMax = dimension;
    }
}