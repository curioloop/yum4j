package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SmootherDiffuseLayout;
import com.curioloop.yum4j.kalman.arena.SmootherResultLayout;
import com.curioloop.yum4j.kalman.internal.ResultArrays;

abstract class SmootherResultBase<T extends SmootherResultBase<T>> implements Cloneable {

    protected static final double[] EMPTY = ResultArrays.emptyArray();

    public int kEndog;
    public int kStates;
    public int kPosdef;
    public int nobs;

    public double[] smoothedState;
    public double[] smoothedStateCov;
    public double[] smoothedObsDisturbance;
    public double[] smoothedStateDisturbance;
    public double[] smoothedObsDisturbanceCov;
    public double[] smoothedStateDisturbanceCov;
    public double[] scaledSmoothedEstimator;
    public double[] scaledSmoothedEstimatorCov;
    public double[] smoothingError;
    public double[] innovationsTransition;

    public double[] scaledSmoothedDiffuseEstimator;
    public double[] scaledSmoothedDiffuse1EstimatorCov;
    public double[] scaledSmoothedDiffuse2EstimatorCov;

    private int smoothedStateBase;
    private int smoothedStateLength;
    private int smoothedStateCovBase;
    private int smoothedStateCovLength;
    private int smoothedObsDisturbanceBase;
    private int smoothedObsDisturbanceLength;
    private int smoothedStateDisturbanceBase;
    private int smoothedStateDisturbanceLength;
    private int smoothedObsDisturbanceCovBase;
    private int smoothedObsDisturbanceCovLength;
    private int smoothedStateDisturbanceCovBase;
    private int smoothedStateDisturbanceCovLength;
    private int scaledSmoothedEstimatorBase;
    private int scaledSmoothedEstimatorLength;
    private int scaledSmoothedEstimatorCovBase;
    private int scaledSmoothedEstimatorCovLength;
    private int smoothingErrorBase;
    private int smoothingErrorLength;
    private int innovationsTransitionBase;
    private int innovationsTransitionLength;
    private int scaledSmoothedDiffuseEstimatorBase;
    private int scaledSmoothedDiffuseEstimatorLength;
    private int scaledSmoothedDiffuse1EstimatorCovBase;
    private int scaledSmoothedDiffuse1EstimatorCovLength;
    private int scaledSmoothedDiffuse2EstimatorCovBase;
    private int scaledSmoothedDiffuse2EstimatorCovLength;

    protected SmootherResultBase() {
        this.kEndog = 0;
        this.kStates = 0;
        this.kPosdef = 0;
        this.nobs = 0;
        this.smoothedState = EMPTY;
        this.smoothedStateCov = EMPTY;
        this.smoothedObsDisturbance = EMPTY;
        this.smoothedStateDisturbance = EMPTY;
        this.smoothedObsDisturbanceCov = EMPTY;
        this.smoothedStateDisturbanceCov = EMPTY;
        this.scaledSmoothedEstimator = EMPTY;
        this.scaledSmoothedEstimatorCov = EMPTY;
        this.smoothingError = EMPTY;
        this.innovationsTransition = EMPTY;
        this.scaledSmoothedDiffuseEstimator = null;
        this.scaledSmoothedDiffuse1EstimatorCov = null;
        this.scaledSmoothedDiffuse2EstimatorCov = null;
    }

    protected SmootherResultBase(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout) {
        this();
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        int stateLength = layout.smoothedStateLength();
        int stateCovLength = layout.smoothedStateCovLength();
        int obsDistLength = layout.smoothedObsDisturbanceLength();
        int stateDistLength = layout.smoothedStateDisturbanceLength();
        int obsDistCovLength = layout.smoothedObsDisturbanceCovLength();
        int stateDistCovLength = layout.smoothedStateDisturbanceCovLength();
        int scaledEstimatorLength = layout.scaledSmoothedEstimatorLength();
        int scaledEstimatorCovLength = layout.scaledSmoothedEstimatorCovLength();
        int smoothingErrorLength = layout.smoothingErrorLength();
        int innovationsTransitionLength = layout.innovationsTransitionLength();
        setSurface(0, stateLength > 0 ? new double[stateLength] : EMPTY, 0, stateLength);
        setSurface(1, stateCovLength > 0 ? new double[stateCovLength] : EMPTY, 0, stateCovLength);
        setSurface(2, obsDistLength > 0 ? new double[obsDistLength] : EMPTY, 0, obsDistLength);
        setSurface(3, stateDistLength > 0 ? new double[stateDistLength] : EMPTY, 0, stateDistLength);
        setSurface(4, obsDistCovLength > 0 ? new double[obsDistCovLength] : EMPTY, 0, obsDistCovLength);
        setSurface(5, stateDistCovLength > 0 ? new double[stateDistCovLength] : EMPTY, 0, stateDistCovLength);
        setSurface(6, scaledEstimatorLength > 0 ? new double[scaledEstimatorLength] : EMPTY, 0, scaledEstimatorLength);
        setSurface(7, scaledEstimatorCovLength > 0 ? new double[scaledEstimatorCovLength] : EMPTY, 0, scaledEstimatorCovLength);
        setSurface(8, smoothingErrorLength > 0 ? new double[smoothingErrorLength] : EMPTY, 0, smoothingErrorLength);
        setSurface(9, innovationsTransitionLength > 0 ? new double[innovationsTransitionLength] : EMPTY, 0, innovationsTransitionLength);
    }

    protected abstract int scalarWidth();

    protected abstract T newResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout);

    @SuppressWarnings("unchecked")
    private T self() {
        return (T) this;
    }

    final T reuse(int kEndog, int kStates, int kPosdef, int nobs,
                  double[] backing,
                  SmootherResultLayout layout) {
        bind(kEndog, kStates, kPosdef, nobs, backing, layout);
        this.scaledSmoothedDiffuseEstimator = null;
        this.scaledSmoothedDiffuse1EstimatorCov = null;
        this.scaledSmoothedDiffuse2EstimatorCov = null;
        return self();
    }

    final T reuseDiffuseAuxiliary(double[] backing, SmootherDiffuseLayout diffuseLayout) {
        setDiffuseSurface(0,
            diffuseLayout.diffuseEstimatorLength() == 0 ? EMPTY : backing,
            diffuseLayout.diffuseEstimatorBase(),
            diffuseLayout.diffuseEstimatorLength());
        setDiffuseSurface(1,
            diffuseLayout.diffuse1EstimatorCovLength() == 0 ? EMPTY : backing,
            diffuseLayout.diffuse1EstimatorCovBase(),
            diffuseLayout.diffuse1EstimatorCovLength());
        setDiffuseSurface(2,
            diffuseLayout.diffuse2EstimatorCovLength() == 0 ? EMPTY : backing,
            diffuseLayout.diffuse2EstimatorCovBase(),
            diffuseLayout.diffuse2EstimatorCovLength());
        return self();
    }

    final T reuseExternal(int kEndog, int kStates, int kPosdef, int nobs,
                          boolean storeState,
                          boolean storeDisturbance,
                          double[] smoothedState,
                          double[] smoothedObsDisturbance,
                          double[] smoothedStateDisturbance) {
        return reuseExternal(kEndog, kStates, kPosdef, nobs,
            storeState,
            storeDisturbance,
            smoothedState,
            0,
            smoothedObsDisturbance,
            0,
            smoothedStateDisturbance,
            0);
    }

    final T reuseExternal(int kEndog, int kStates, int kPosdef, int nobs,
                          boolean storeState,
                          boolean storeDisturbance,
                          double[] smoothedState,
                          int smoothedStateBase,
                          double[] smoothedObsDisturbance,
                          int smoothedObsDisturbanceBase,
                          double[] smoothedStateDisturbance,
                          int smoothedStateDisturbanceBase) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        int stateLength = storeState ? scalarWidth() * kStates * nobs : 0;
        int obsDisturbanceLength = storeDisturbance ? scalarWidth() * kEndog * nobs : 0;
        int stateDisturbanceLength = storeDisturbance ? scalarWidth() * kPosdef * nobs : 0;
        setSurface(0, storeState ? requireExternalSurface(smoothedState, smoothedStateBase, stateLength, "smoothedState") : EMPTY,
            storeState ? smoothedStateBase : 0, stateLength);
        setSurface(1, EMPTY, 0, 0);
        setSurface(2, storeDisturbance ? requireExternalSurface(smoothedObsDisturbance, smoothedObsDisturbanceBase, obsDisturbanceLength, "smoothedObsDisturbance") : EMPTY,
            storeDisturbance ? smoothedObsDisturbanceBase : 0, obsDisturbanceLength);
        setSurface(3, storeDisturbance ? requireExternalSurface(smoothedStateDisturbance, smoothedStateDisturbanceBase, stateDisturbanceLength, "smoothedStateDisturbance") : EMPTY,
            storeDisturbance ? smoothedStateDisturbanceBase : 0, stateDisturbanceLength);
        setSurface(4, EMPTY, 0, 0);
        setSurface(5, EMPTY, 0, 0);
        setSurface(6, EMPTY, 0, 0);
        setSurface(7, EMPTY, 0, 0);
        setSurface(8, EMPTY, 0, 0);
        setSurface(9, EMPTY, 0, 0);
        setDiffuseSurface(0, null, 0, 0);
        setDiffuseSurface(1, null, 0, 0);
        setDiffuseSurface(2, null, 0, 0);
        return self();
    }

    public final T copy() {
        T copy = newResult(kEndog, kStates, kPosdef, nobs, currentLayout());
        ResultArrays.copySurface(smoothedState, smoothedStateBase(), smoothedStateLength(), copy.smoothedState, copy.smoothedStateBase());
        ResultArrays.copySurface(smoothedStateCov, smoothedStateCovBase(), smoothedStateCovLength(), copy.smoothedStateCov, copy.smoothedStateCovBase());
        ResultArrays.copySurface(smoothedObsDisturbance, smoothedObsDisturbanceBase(), smoothedObsDisturbanceLength(), copy.smoothedObsDisturbance, copy.smoothedObsDisturbanceBase());
        ResultArrays.copySurface(smoothedStateDisturbance, smoothedStateDisturbanceBase(), smoothedStateDisturbanceLength(), copy.smoothedStateDisturbance, copy.smoothedStateDisturbanceBase());
        ResultArrays.copySurface(smoothedObsDisturbanceCov, smoothedObsDisturbanceCovBase(), smoothedObsDisturbanceCovLength(), copy.smoothedObsDisturbanceCov, copy.smoothedObsDisturbanceCovBase());
        ResultArrays.copySurface(smoothedStateDisturbanceCov, smoothedStateDisturbanceCovBase(), smoothedStateDisturbanceCovLength(), copy.smoothedStateDisturbanceCov, copy.smoothedStateDisturbanceCovBase());
        ResultArrays.copySurface(scaledSmoothedEstimator, scaledSmoothedEstimatorBase(), scaledSmoothedEstimatorLength(), copy.scaledSmoothedEstimator, copy.scaledSmoothedEstimatorBase());
        ResultArrays.copySurface(scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovBase(), scaledSmoothedEstimatorCovLength(), copy.scaledSmoothedEstimatorCov, copy.scaledSmoothedEstimatorCovBase());
        ResultArrays.copySurface(smoothingError, smoothingErrorBase(), smoothingErrorLength(), copy.smoothingError, copy.smoothingErrorBase());
        ResultArrays.copySurface(innovationsTransition, innovationsTransitionBase(), innovationsTransitionLength(), copy.innovationsTransition, copy.innovationsTransitionBase());
        if (scaledSmoothedDiffuseEstimator != null || scaledSmoothedDiffuse1EstimatorCov != null || scaledSmoothedDiffuse2EstimatorCov != null) {
            copy.allocateDiffuse();
            ResultArrays.copySurface(scaledSmoothedDiffuseEstimator, scaledSmoothedDiffuseEstimatorBase(), scaledSmoothedDiffuseEstimatorLength(),
                copy.scaledSmoothedDiffuseEstimator, copy.scaledSmoothedDiffuseEstimatorBase());
            ResultArrays.copySurface(scaledSmoothedDiffuse1EstimatorCov, scaledSmoothedDiffuse1EstimatorCovBase(), scaledSmoothedDiffuse1EstimatorCovLength(),
                copy.scaledSmoothedDiffuse1EstimatorCov, copy.scaledSmoothedDiffuse1EstimatorCovBase());
            ResultArrays.copySurface(scaledSmoothedDiffuse2EstimatorCov, scaledSmoothedDiffuse2EstimatorCovBase(), scaledSmoothedDiffuse2EstimatorCovLength(),
                copy.scaledSmoothedDiffuse2EstimatorCov, copy.scaledSmoothedDiffuse2EstimatorCovBase());
            if (scaledSmoothedDiffuseEstimatorLength() == 0) {
                copy.setDiffuseSurface(0, EMPTY, 0, 0);
            }
            if (scaledSmoothedDiffuse1EstimatorCovLength() == 0) {
                copy.setDiffuseSurface(1, EMPTY, 0, 0);
            }
            if (scaledSmoothedDiffuse2EstimatorCovLength() == 0) {
                copy.setDiffuseSurface(2, EMPTY, 0, 0);
            }
        }
        return copy;
    }

    @Override
    public final T clone() {
        return copy();
    }

    public final SmootherResultShape resultShape() {
        return SmootherResultShape.of(
            smoothedStateLength() > 0,
            smoothedStateCovLength() > 0,
            smoothedObsDisturbanceLength() > 0 || smoothedStateDisturbanceLength() > 0,
            smoothedObsDisturbanceCovLength() > 0 || smoothedStateDisturbanceCovLength() > 0,
            scaledSmoothedEstimatorLength() > 0 || scaledSmoothedEstimatorCovLength() > 0
                || smoothingErrorLength() > 0 || innovationsTransitionLength() > 0);
    }

    private SmootherResultLayout currentLayout() {
        return SmootherResultLayout.create(
            scalarWidth(),
            kEndog,
            kStates,
            kPosdef,
            nobs,
            smoothedStateLength() > 0,
            smoothedStateCovLength() > 0,
            smoothedObsDisturbanceLength() > 0 || smoothedStateDisturbanceLength() > 0,
            smoothedObsDisturbanceCovLength() > 0 || smoothedStateDisturbanceCovLength() > 0,
            scaledSmoothedEstimatorLength() > 0 || scaledSmoothedEstimatorCovLength() > 0
                || smoothingErrorLength() > 0 || innovationsTransitionLength() > 0);
    }

    public final void allocateDiffuse() {
        if (scaledSmoothedDiffuseEstimator != null) {
            return;
        }
        int scalarWidth = scalarWidth();
        int estimatorLength = scalarWidth * kStates * (nobs + 1);
        int covarianceLength = kStates * scalarWidth * kStates * (nobs + 1);
        setDiffuseSurface(0, new double[estimatorLength], 0, estimatorLength);
        setDiffuseSurface(1, new double[covarianceLength], 0, covarianceLength);
        setDiffuseSurface(2, new double[covarianceLength], 0, covarianceLength);
    }

    public final void suppressState() {
        setSurface(0, EMPTY, 0, 0);
    }

    public final void suppressStateCovariance() {
        setSurface(1, EMPTY, 0, 0);
    }

    public final void suppressDisturbance() {
        setSurface(2, EMPTY, 0, 0);
        setSurface(3, EMPTY, 0, 0);
    }

    public final void suppressDisturbanceCovariance() {
        setSurface(4, EMPTY, 0, 0);
        setSurface(5, EMPTY, 0, 0);
    }

    public final void suppressAuxiliary() {
        setSurface(6, EMPTY, 0, 0);
        setSurface(7, EMPTY, 0, 0);
        setSurface(8, EMPTY, 0, 0);
        setSurface(9, EMPTY, 0, 0);
    }

    public final void suppressDiffuseAuxiliary() {
        setDiffuseSurface(0, EMPTY, 0, 0);
        setDiffuseSurface(1, EMPTY, 0, 0);
        setDiffuseSurface(2, EMPTY, 0, 0);
    }

    final T trimOutputMask(int smootherOutput) {
        if ((smootherOutput & KalmanSmoother.SMOOTHER_STATE) == 0) {
            suppressState();
        }
        if ((smootherOutput & KalmanSmoother.SMOOTHER_STATE_COV) == 0) {
            suppressStateCovariance();
        }
        if ((smootherOutput & KalmanSmoother.SMOOTHER_DISTURBANCE) == 0) {
            suppressDisturbance();
        }
        if ((smootherOutput & KalmanSmoother.SMOOTHER_DISTURBANCE_COV) == 0) {
            suppressDisturbanceCovariance();
        }
        if (smootherOutput != KalmanSmoother.SMOOTHER_ALL) {
            suppressAuxiliary();
            suppressDiffuseAuxiliary();
        }
        return self();
    }

    public final int smoothedStateBase() {
        return smoothedStateBase;
    }

    public final int smoothedStateLength() {
        return smoothedStateLength;
    }

    public final int smoothedStateOffset(int t) {
        return smoothedStateBase() + t * scalarWidth() * kStates;
    }

    public final int smoothedStateCovBase() {
        return smoothedStateCovBase;
    }

    public final int smoothedStateCovLength() {
        return smoothedStateCovLength;
    }

    public final int smoothedStateCovOffset(int t) {
        return smoothedStateCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int smoothedObsDisturbanceBase() {
        return smoothedObsDisturbanceBase;
    }

    public final int smoothedObsDisturbanceLength() {
        return smoothedObsDisturbanceLength;
    }

    public final int smoothedObsDisturbanceOffset(int t) {
        return smoothedObsDisturbanceBase() + t * scalarWidth() * kEndog;
    }

    public final int smoothedStateDisturbanceBase() {
        return smoothedStateDisturbanceBase;
    }

    public final int smoothedStateDisturbanceLength() {
        return smoothedStateDisturbanceLength;
    }

    public final int smoothedStateDisturbanceOffset(int t) {
        return smoothedStateDisturbanceBase() + t * scalarWidth() * kPosdef;
    }

    public final int smoothedObsDisturbanceCovBase() {
        return smoothedObsDisturbanceCovBase;
    }

    public final int smoothedObsDisturbanceCovLength() {
        return smoothedObsDisturbanceCovLength;
    }

    public final int smoothedObsDisturbanceCovOffset(int t) {
        return smoothedObsDisturbanceCovBase() + t * kEndog * scalarWidth() * kEndog;
    }

    public final int smoothedStateDisturbanceCovBase() {
        return smoothedStateDisturbanceCovBase;
    }

    public final int smoothedStateDisturbanceCovLength() {
        return smoothedStateDisturbanceCovLength;
    }

    public final int smoothedStateDisturbanceCovOffset(int t) {
        return smoothedStateDisturbanceCovBase() + t * kPosdef * scalarWidth() * kPosdef;
    }

    public final int scaledSmoothedEstimatorBase() {
        return scaledSmoothedEstimatorBase;
    }

    public final int scaledSmoothedEstimatorLength() {
        return scaledSmoothedEstimatorLength;
    }

    public final int scaledSmoothedEstimatorOffset(int t) {
        return scaledSmoothedEstimatorBase() + t * scalarWidth() * kStates;
    }

    public final int scaledSmoothedEstimatorCovBase() {
        return scaledSmoothedEstimatorCovBase;
    }

    public final int scaledSmoothedEstimatorCovLength() {
        return scaledSmoothedEstimatorCovLength;
    }

    public final int scaledSmoothedEstimatorCovOffset(int t) {
        return scaledSmoothedEstimatorCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int smoothingErrorBase() {
        return smoothingErrorBase;
    }

    public final int smoothingErrorLength() {
        return smoothingErrorLength;
    }

    public final int smoothingErrorOffset(int t) {
        return smoothingErrorBase() + t * scalarWidth() * kEndog;
    }

    public final int innovationsTransitionBase() {
        return innovationsTransitionBase;
    }

    public final int innovationsTransitionLength() {
        return innovationsTransitionLength;
    }

    public final int innovationsTransitionOffset(int t) {
        return innovationsTransitionBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int scaledSmoothedDiffuseEstimatorBase() {
        return scaledSmoothedDiffuseEstimatorBase;
    }

    public final int scaledSmoothedDiffuseEstimatorLength() {
        return scaledSmoothedDiffuseEstimatorLength;
    }

    public final int scaledSmoothedDiffuseEstimatorOffset(int t) {
        return scaledSmoothedDiffuseEstimatorBase() + t * scalarWidth() * kStates;
    }

    public final int scaledSmoothedDiffuse1EstimatorCovBase() {
        return scaledSmoothedDiffuse1EstimatorCovBase;
    }

    public final int scaledSmoothedDiffuse1EstimatorCovLength() {
        return scaledSmoothedDiffuse1EstimatorCovLength;
    }

    public final int scaledSmoothedDiffuse1EstimatorCovOffset(int t) {
        return scaledSmoothedDiffuse1EstimatorCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int scaledSmoothedDiffuse2EstimatorCovBase() {
        return scaledSmoothedDiffuse2EstimatorCovBase;
    }

    public final int scaledSmoothedDiffuse2EstimatorCovLength() {
        return scaledSmoothedDiffuse2EstimatorCovLength;
    }

    public final int scaledSmoothedDiffuse2EstimatorCovOffset(int t) {
        return scaledSmoothedDiffuse2EstimatorCovBase() + t * kStates * scalarWidth() * kStates;
    }

    private void bind(int kEndog, int kStates, int kPosdef, int nobs,
                      double[] backing,
                      SmootherResultLayout layout) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        setSurface(0, layout.smoothedStateLength() == 0 ? EMPTY : backing, layout.smoothedStateBase(), layout.smoothedStateLength());
        setSurface(1, layout.smoothedStateCovLength() == 0 ? EMPTY : backing, layout.smoothedStateCovBase(), layout.smoothedStateCovLength());
        setSurface(2, layout.smoothedObsDisturbanceLength() == 0 ? EMPTY : backing, layout.smoothedObsDisturbanceBase(), layout.smoothedObsDisturbanceLength());
        setSurface(3, layout.smoothedStateDisturbanceLength() == 0 ? EMPTY : backing, layout.smoothedStateDisturbanceBase(), layout.smoothedStateDisturbanceLength());
        setSurface(4, layout.smoothedObsDisturbanceCovLength() == 0 ? EMPTY : backing, layout.smoothedObsDisturbanceCovBase(), layout.smoothedObsDisturbanceCovLength());
        setSurface(5, layout.smoothedStateDisturbanceCovLength() == 0 ? EMPTY : backing, layout.smoothedStateDisturbanceCovBase(), layout.smoothedStateDisturbanceCovLength());
        setSurface(6, layout.scaledSmoothedEstimatorLength() == 0 ? EMPTY : backing, layout.scaledSmoothedEstimatorBase(), layout.scaledSmoothedEstimatorLength());
        setSurface(7, layout.scaledSmoothedEstimatorCovLength() == 0 ? EMPTY : backing, layout.scaledSmoothedEstimatorCovBase(), layout.scaledSmoothedEstimatorCovLength());
        setSurface(8, layout.smoothingErrorLength() == 0 ? EMPTY : backing, layout.smoothingErrorBase(), layout.smoothingErrorLength());
        setSurface(9, layout.innovationsTransitionLength() == 0 ? EMPTY : backing, layout.innovationsTransitionBase(), layout.innovationsTransitionLength());
    }

    private static double[] requireExternalSurface(double[] surface, int base, int expectedLength, String name) {
        if (surface == null || base < 0 || surface.length < base + expectedLength) {
            throw new IllegalArgumentException(name + " span must fit length " + expectedLength + " at base " + base);
        }
        return surface;
    }

    final void setSurface(int index, double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        switch (index) {
            case 0 -> {
                smoothedState = assigned;
                smoothedStateBase = assignedBase;
                smoothedStateLength = assignedLength;
            }
            case 1 -> {
                smoothedStateCov = assigned;
                smoothedStateCovBase = assignedBase;
                smoothedStateCovLength = assignedLength;
            }
            case 2 -> {
                smoothedObsDisturbance = assigned;
                smoothedObsDisturbanceBase = assignedBase;
                smoothedObsDisturbanceLength = assignedLength;
            }
            case 3 -> {
                smoothedStateDisturbance = assigned;
                smoothedStateDisturbanceBase = assignedBase;
                smoothedStateDisturbanceLength = assignedLength;
            }
            case 4 -> {
                smoothedObsDisturbanceCov = assigned;
                smoothedObsDisturbanceCovBase = assignedBase;
                smoothedObsDisturbanceCovLength = assignedLength;
            }
            case 5 -> {
                smoothedStateDisturbanceCov = assigned;
                smoothedStateDisturbanceCovBase = assignedBase;
                smoothedStateDisturbanceCovLength = assignedLength;
            }
            case 6 -> {
                scaledSmoothedEstimator = assigned;
                scaledSmoothedEstimatorBase = assignedBase;
                scaledSmoothedEstimatorLength = assignedLength;
            }
            case 7 -> {
                scaledSmoothedEstimatorCov = assigned;
                scaledSmoothedEstimatorCovBase = assignedBase;
                scaledSmoothedEstimatorCovLength = assignedLength;
            }
            case 8 -> {
                smoothingError = assigned;
                smoothingErrorBase = assignedBase;
                smoothingErrorLength = assignedLength;
            }
            case 9 -> {
                innovationsTransition = assigned;
                innovationsTransitionBase = assignedBase;
                innovationsTransitionLength = assignedLength;
            }
            default -> throw new IllegalArgumentException("Unexpected surface index " + index);
        }
    }

    final void setDiffuseSurface(int index, double[] surface, int base, int length) {
        double[] assigned = surface;
        int assignedBase = assigned == null || assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == null || assigned == EMPTY ? 0 : length;
        switch (index) {
            case 0 -> {
                scaledSmoothedDiffuseEstimator = assigned;
                scaledSmoothedDiffuseEstimatorBase = assignedBase;
                scaledSmoothedDiffuseEstimatorLength = assignedLength;
            }
            case 1 -> {
                scaledSmoothedDiffuse1EstimatorCov = assigned;
                scaledSmoothedDiffuse1EstimatorCovBase = assignedBase;
                scaledSmoothedDiffuse1EstimatorCovLength = assignedLength;
            }
            case 2 -> {
                scaledSmoothedDiffuse2EstimatorCov = assigned;
                scaledSmoothedDiffuse2EstimatorCovBase = assignedBase;
                scaledSmoothedDiffuse2EstimatorCovLength = assignedLength;
            }
            default -> throw new IllegalArgumentException("Unexpected diffuse surface index " + index);
        }
    }
}