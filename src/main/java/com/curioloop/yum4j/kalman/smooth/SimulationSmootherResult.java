package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SimulationResultLayout;
import com.curioloop.yum4j.kalman.internal.ResultArrays;

/**
 * Flat storage for generated and simulation-smoothed draws.
 */
public final class SimulationSmootherResult implements Cloneable {

    private static final double[] EMPTY = ResultArrays.emptyArray();

    public int kEndog;
    public int kStates;
    public int kPosdef;
    public int nobs;

    public double[] generatedMeasurementDisturbance;
    public double[] generatedStateDisturbance;
    public double[] generatedObs;
    public double[] generatedState;

    public double[] simulatedMeasurementDisturbance;
    public double[] simulatedStateDisturbance;
    public double[] simulatedState;

    private int generatedMeasurementDisturbanceBase;
    private int generatedMeasurementDisturbanceLength;
    private int generatedStateDisturbanceBase;
    private int generatedStateDisturbanceLength;
    private int generatedObsBase;
    private int generatedObsLength;
    private int generatedStateBase;
    private int generatedStateLength;
    private int simulatedMeasurementDisturbanceBase;
    private int simulatedMeasurementDisturbanceLength;
    private int simulatedStateDisturbanceBase;
    private int simulatedStateDisturbanceLength;
    private int simulatedStateBase;
    private int simulatedStateLength;

    SimulationSmootherResult() {
        this.kEndog = 0;
        this.kStates = 0;
        this.kPosdef = 0;
        this.nobs = 0;
        this.generatedMeasurementDisturbance = EMPTY;
        this.generatedStateDisturbance = EMPTY;
        this.generatedObs = EMPTY;
        this.generatedState = EMPTY;
        this.simulatedMeasurementDisturbance = EMPTY;
        this.simulatedStateDisturbance = EMPTY;
        this.simulatedState = EMPTY;
    }

    public SimulationSmootherResult(int kEndog, int kStates, int kPosdef, int nobs,
                                    boolean storeState, boolean storeDisturbance) {
        this(kEndog, kStates, kPosdef, nobs, true, storeState, storeDisturbance);
    }

    public SimulationSmootherResult(int kEndog, int kStates, int kPosdef, int nobs,
                                    boolean storeGenerated,
                                    boolean storeState,
                                    boolean storeDisturbance) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        int generatedMeasurementLength = storeGenerated ? kEndog * nobs : 0;
        int generatedStateDisturbanceLength = storeGenerated ? kPosdef * nobs : 0;
        int generatedObsLength = storeGenerated ? kEndog * nobs : 0;
        int generatedStateLength = storeGenerated ? kStates * (nobs + 1) : 0;
        int simulatedMeasurementLength = storeDisturbance ? kEndog * nobs : 0;
        int simulatedStateDisturbanceLength = storeDisturbance ? kPosdef * nobs : 0;
        int simulatedStateLength = storeState ? kStates * nobs : 0;
        setSurface(0, generatedMeasurementLength > 0 ? ResultArrays.allocate(generatedMeasurementLength) : EMPTY, 0, generatedMeasurementLength);
        setSurface(1, generatedStateDisturbanceLength > 0 ? ResultArrays.allocate(generatedStateDisturbanceLength) : EMPTY, 0, generatedStateDisturbanceLength);
        setSurface(2, generatedObsLength > 0 ? ResultArrays.allocate(generatedObsLength) : EMPTY, 0, generatedObsLength);
        setSurface(3, generatedStateLength > 0 ? ResultArrays.allocate(generatedStateLength) : EMPTY, 0, generatedStateLength);
        setSurface(4, simulatedMeasurementLength > 0 ? ResultArrays.allocate(simulatedMeasurementLength) : EMPTY, 0, simulatedMeasurementLength);
        setSurface(5, simulatedStateDisturbanceLength > 0 ? ResultArrays.allocate(simulatedStateDisturbanceLength) : EMPTY, 0, simulatedStateDisturbanceLength);
        setSurface(6, simulatedStateLength > 0 ? ResultArrays.allocate(simulatedStateLength) : EMPTY, 0, simulatedStateLength);
    }

    static double[] emptyArray() {
        return EMPTY;
    }

    SimulationSmootherResult reuse(int kEndog, int kStates, int kPosdef, int nobs,
                                   double[] backing,
                                   SimulationResultLayout layout) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        setSurface(0, layout.generatedMeasurementDisturbanceLength() == 0 ? EMPTY : backing,
            layout.generatedMeasurementDisturbanceBase(), layout.generatedMeasurementDisturbanceLength());
        setSurface(1, layout.generatedStateDisturbanceLength() == 0 ? EMPTY : backing,
            layout.generatedStateDisturbanceBase(), layout.generatedStateDisturbanceLength());
        setSurface(2, layout.generatedObsLength() == 0 ? EMPTY : backing,
            layout.generatedObsBase(), layout.generatedObsLength());
        setSurface(3, layout.generatedStateLength() == 0 ? EMPTY : backing,
            layout.generatedStateBase(), layout.generatedStateLength());
        setSurface(4, layout.simulatedMeasurementDisturbanceLength() == 0 ? EMPTY : backing,
            layout.simulatedMeasurementDisturbanceBase(), layout.simulatedMeasurementDisturbanceLength());
        setSurface(5, layout.simulatedStateDisturbanceLength() == 0 ? EMPTY : backing,
            layout.simulatedStateDisturbanceBase(), layout.simulatedStateDisturbanceLength());
        setSurface(6, layout.simulatedStateLength() == 0 ? EMPTY : backing,
            layout.simulatedStateBase(), layout.simulatedStateLength());
        return this;
    }

    public SimulationSmootherResult copy() {
        boolean storeGenerated = generatedMeasurementDisturbanceLength() > 0
            || generatedStateDisturbanceLength() > 0
            || generatedObsLength() > 0
            || generatedStateLength() > 0;
        boolean storeState = simulatedStateLength() > 0;
        boolean storeDisturbance = simulatedMeasurementDisturbanceLength() > 0 || simulatedStateDisturbanceLength() > 0;
        SimulationSmootherResult copy = new SimulationSmootherResult(kEndog, kStates, kPosdef, nobs,
            storeGenerated, storeState, storeDisturbance);
        ResultArrays.copySurface(generatedMeasurementDisturbance, generatedMeasurementDisturbanceBase(), generatedMeasurementDisturbanceLength(),
            copy.generatedMeasurementDisturbance, copy.generatedMeasurementDisturbanceBase());
        ResultArrays.copySurface(generatedStateDisturbance, generatedStateDisturbanceBase(), generatedStateDisturbanceLength(),
            copy.generatedStateDisturbance, copy.generatedStateDisturbanceBase());
        ResultArrays.copySurface(generatedObs, generatedObsBase(), generatedObsLength(), copy.generatedObs, copy.generatedObsBase());
        ResultArrays.copySurface(generatedState, generatedStateBase(), generatedStateLength(), copy.generatedState, copy.generatedStateBase());
        ResultArrays.copySurface(simulatedMeasurementDisturbance, simulatedMeasurementDisturbanceBase(), simulatedMeasurementDisturbanceLength(),
            copy.simulatedMeasurementDisturbance, copy.simulatedMeasurementDisturbanceBase());
        ResultArrays.copySurface(simulatedStateDisturbance, simulatedStateDisturbanceBase(), simulatedStateDisturbanceLength(),
            copy.simulatedStateDisturbance, copy.simulatedStateDisturbanceBase());
        ResultArrays.copySurface(simulatedState, simulatedStateBase(), simulatedStateLength(), copy.simulatedState, copy.simulatedStateBase());
        return copy;
    }

    @Override
    public SimulationSmootherResult clone() {
        return copy();
    }

    public double generatedMeasurementDisturbance(int i, int t) {
        if (generatedMeasurementDisturbanceLength() == 0) {
            return 0.0;
        }
        return generatedMeasurementDisturbance[generatedMeasurementDisturbanceOffset(t) + i];
    }

    public double generatedStateDisturbance(int i, int t) {
        if (generatedStateDisturbanceLength() == 0) {
            return 0.0;
        }
        return generatedStateDisturbance[generatedStateDisturbanceOffset(t) + i];
    }

    public double generatedObs(int i, int t) {
        if (generatedObsLength() == 0) {
            return 0.0;
        }
        return generatedObs[generatedObsOffset(t) + i];
    }

    public double generatedState(int i, int t) {
        if (generatedStateLength() == 0) {
            return 0.0;
        }
        return generatedState[generatedStateOffset(t) + i];
    }

    public double simulatedMeasurementDisturbance(int i, int t) {
        if (simulatedMeasurementDisturbanceLength() == 0) {
            return 0.0;
        }
        return simulatedMeasurementDisturbance[simulatedMeasurementDisturbanceOffset(t) + i];
    }

    public double simulatedStateDisturbance(int i, int t) {
        if (simulatedStateDisturbanceLength() == 0) {
            return 0.0;
        }
        return simulatedStateDisturbance[simulatedStateDisturbanceOffset(t) + i];
    }

    public double simulatedState(int i, int t) {
        if (simulatedStateLength() == 0) {
            return 0.0;
        }
        return simulatedState[simulatedStateOffset(t) + i];
    }

    public int generatedMeasurementDisturbanceBase() {
        return generatedMeasurementDisturbanceBase;
    }

    public int generatedMeasurementDisturbanceLength() {
        return generatedMeasurementDisturbanceLength;
    }

    public int generatedMeasurementDisturbanceOffset(int t) {
        return generatedMeasurementDisturbanceBase() + t * kEndog;
    }

    public int generatedStateDisturbanceBase() {
        return generatedStateDisturbanceBase;
    }

    public int generatedStateDisturbanceLength() {
        return generatedStateDisturbanceLength;
    }

    public int generatedStateDisturbanceOffset(int t) {
        return generatedStateDisturbanceBase() + t * kPosdef;
    }

    public int generatedObsBase() {
        return generatedObsBase;
    }

    public int generatedObsLength() {
        return generatedObsLength;
    }

    public int generatedObsOffset(int t) {
        return generatedObsBase() + t * kEndog;
    }

    public int generatedStateBase() {
        return generatedStateBase;
    }

    public int generatedStateLength() {
        return generatedStateLength;
    }

    public int generatedStateOffset(int t) {
        return generatedStateBase() + t * kStates;
    }

    public int simulatedMeasurementDisturbanceBase() {
        return simulatedMeasurementDisturbanceBase;
    }

    public int simulatedMeasurementDisturbanceLength() {
        return simulatedMeasurementDisturbanceLength;
    }

    public int simulatedMeasurementDisturbanceOffset(int t) {
        return simulatedMeasurementDisturbanceBase() + t * kEndog;
    }

    public int simulatedStateDisturbanceBase() {
        return simulatedStateDisturbanceBase;
    }

    public int simulatedStateDisturbanceLength() {
        return simulatedStateDisturbanceLength;
    }

    public int simulatedStateDisturbanceOffset(int t) {
        return simulatedStateDisturbanceBase() + t * kPosdef;
    }

    public int simulatedStateBase() {
        return simulatedStateBase;
    }

    public int simulatedStateLength() {
        return simulatedStateLength;
    }

    public int simulatedStateOffset(int t) {
        return simulatedStateBase() + t * kStates;
    }

    private void setSurface(int index, double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        switch (index) {
            case 0 -> {
                generatedMeasurementDisturbance = assigned;
                generatedMeasurementDisturbanceBase = assignedBase;
                generatedMeasurementDisturbanceLength = assignedLength;
            }
            case 1 -> {
                generatedStateDisturbance = assigned;
                generatedStateDisturbanceBase = assignedBase;
                generatedStateDisturbanceLength = assignedLength;
            }
            case 2 -> {
                generatedObs = assigned;
                generatedObsBase = assignedBase;
                generatedObsLength = assignedLength;
            }
            case 3 -> {
                generatedState = assigned;
                generatedStateBase = assignedBase;
                generatedStateLength = assignedLength;
            }
            case 4 -> {
                simulatedMeasurementDisturbance = assigned;
                simulatedMeasurementDisturbanceBase = assignedBase;
                simulatedMeasurementDisturbanceLength = assignedLength;
            }
            case 5 -> {
                simulatedStateDisturbance = assigned;
                simulatedStateDisturbanceBase = assignedBase;
                simulatedStateDisturbanceLength = assignedLength;
            }
            case 6 -> {
                simulatedState = assigned;
                simulatedStateBase = assignedBase;
                simulatedStateLength = assignedLength;
            }
            default -> throw new IllegalArgumentException("Unexpected surface index " + index);
        }
    }

}