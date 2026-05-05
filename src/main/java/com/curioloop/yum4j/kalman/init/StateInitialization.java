package com.curioloop.yum4j.kalman.init;

import com.curioloop.yum4j.kalman.model.StateSpaceModel;

import java.util.Arrays;
import java.util.Objects;

/**
 * Initialization payload for one Kalman filter run.
 *
 * <p>The state vector and covariance use the same flat layout as the target
 * filter implementation: real values for the real filter and interleaved
 * complex values for the complex filter.</p>
 */
public final class StateInitialization {

    private static final int UNSPECIFIED_DIMENSION = -1;
    private static final double DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE = 1e6;

    private enum Kind {
        KNOWN,
        DIFFUSE,
        APPROXIMATE_DIFFUSE,
        STATIONARY,
        COMPOSITE,
        RESOLVED
    }

    private final Kind kind;
    private final int dimension;
    private final double approximateDiffuseVariance;
    private final double[] initialState;
    private final double[] initialStateCov;
    private final double[] initialDiffuseStateCov;

    private int[] blockStarts;
    private int[] blockEnds;
    private StateInitialization[] blockInitializations;
    private int blockCount;

    public static final class StationaryWorkspace {
        private double[] backing;
        private int[] pivots;

        public void ensureCapacity(StateSpaceModel model) {
            Objects.requireNonNull(model, "model");
            int requiredBacking = requiredStationaryBackingLength(model);
            if (backing == null || backing.length < requiredBacking) {
                backing = new double[requiredBacking];
            }
            int requiredPivots = requiredStationaryPivotLength(model);
            if (pivots == null || pivots.length < requiredPivots) {
                pivots = new int[requiredPivots];
            }
        }

        public void release() {
            backing = null;
            pivots = null;
        }

        public long retainedDoubleCount() {
            long backingCount = backing == null ? 0L : backing.length;
            long pivotCount = pivots == null ? 0L : pivots.length * ((long) Integer.BYTES) / Double.BYTES;
            return backingCount + pivotCount;
        }
    }

    public StateInitialization(int dimension) {
        this(Kind.COMPOSITE, dimension, null, null, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
        validateCompositeDimension(dimension);
    }

    private StateInitialization(Kind kind,
                                int dimension,
                                double[] initialState,
                                double[] initialStateCov,
                                double[] initialDiffuseStateCov,
                                double approximateDiffuseVariance) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.dimension = dimension;
        this.initialState = initialState;
        this.initialStateCov = initialStateCov;
        this.initialDiffuseStateCov = initialDiffuseStateCov;
        this.approximateDiffuseVariance = approximateDiffuseVariance;
    }

    public static StateInitialization known(double[] initialState, double[] initialStateCov) {
        validateKnownInputs(initialState, initialStateCov);
        return new StateInitialization(Kind.KNOWN, UNSPECIFIED_DIMENSION,
            initialState, initialStateCov, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization known(int dimension, double[] initialState, double[] initialStateCov) {
        validateCompositeDimension(dimension);
        validateKnownInputs(initialState, initialStateCov);
        return new StateInitialization(Kind.KNOWN, dimension,
            initialState, initialStateCov, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization diffuse() {
        return new StateInitialization(Kind.DIFFUSE, UNSPECIFIED_DIMENSION,
            null, null, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization diffuse(int dimension) {
        validateCompositeDimension(dimension);
        return new StateInitialization(Kind.DIFFUSE, dimension,
            null, null, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization approximateDiffuse() {
        return approximateDiffuse(UNSPECIFIED_DIMENSION, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization approximateDiffuse(double approximateDiffuseVariance) {
        return approximateDiffuse(UNSPECIFIED_DIMENSION, null, approximateDiffuseVariance);
    }

    public static StateInitialization approximateDiffuse(int dimension) {
        return approximateDiffuse(dimension, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization approximateDiffuse(int dimension, double approximateDiffuseVariance) {
        return approximateDiffuse(dimension, null, approximateDiffuseVariance);
    }

    public static StateInitialization approximateDiffuse(double[] initialState, double approximateDiffuseVariance) {
        return approximateDiffuse(UNSPECIFIED_DIMENSION, initialState, approximateDiffuseVariance);
    }

    public static StateInitialization approximateDiffuse(int dimension,
                                                         double[] initialState,
                                                         double approximateDiffuseVariance) {
        validateApproximateDiffuseVariance(approximateDiffuseVariance);
        if (dimension != UNSPECIFIED_DIMENSION) {
            validateCompositeDimension(dimension);
        }
        return new StateInitialization(Kind.APPROXIMATE_DIFFUSE, dimension,
            initialState, null, null, approximateDiffuseVariance);
    }

    public static StateInitialization stationary(double[] initialState, double[] initialStateCov) {
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(initialStateCov, "initialStateCov");
        return new StateInitialization(Kind.STATIONARY, UNSPECIFIED_DIMENSION,
            initialState, initialStateCov, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization stationary(int dimension) {
        validateCompositeDimension(dimension);
        return new StateInitialization(Kind.STATIONARY, dimension,
            null, null, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    public static StateInitialization stationary(StateSpaceModel model) {
        return stationary(model, 0);
    }

    public static StateInitialization stationary(StateSpaceModel model, int t) {
        Objects.requireNonNull(model, "model");
        validateTimeIndex(model.observationCount(), t);
        StateInitialization solution = solveStationary(model, t);
        if (solution == null) {
            throw new IllegalArgumentException("State block is not discrete-stationary at t=" + t);
        }
        return solution;
    }

    public StateInitialization set(StateInitialization initialization) {
        return set(0, dimension, initialization);
    }

    public StateInitialization set(int stateIndex, StateInitialization initialization) {
        return set(stateIndex, stateIndex + 1, initialization);
    }

    public StateInitialization set(int startInclusive, int endExclusive, StateInitialization initialization) {
        requireComposite();
        Objects.requireNonNull(initialization, "initialization");
        validateRange(startInclusive, endExclusive, dimension);
        validateBlockDimension(initialization, endExclusive - startInclusive);
        insertBlock(startInclusive, endExclusive, initialization);
        return this;
    }

    public StateInitialization unset(int stateIndex) {
        return unset(stateIndex, stateIndex + 1);
    }

    public StateInitialization unset(int startInclusive, int endExclusive) {
        requireComposite();
        validateRange(startInclusive, endExclusive, dimension);
        int index = findExactBlock(startInclusive, endExclusive);
        if (index < 0) {
            throw new IllegalArgumentException("No initialization block set for [" + startInclusive + ", " + endExclusive + ")");
        }
        removeBlock(index);
        return this;
    }

    public StateInitialization clear() {
        requireComposite();
        blockCount = 0;
        return this;
    }

    public boolean isKnown() {
        return kind == Kind.KNOWN;
    }

    public boolean isDiffuse() {
        return kind == Kind.DIFFUSE;
    }

    public boolean isApproximateDiffuse() {
        return kind == Kind.APPROXIMATE_DIFFUSE;
    }

    public boolean isStationary() {
        return kind == Kind.STATIONARY;
    }

    public boolean hasDiffuseComponent() {
        if (kind == Kind.DIFFUSE) {
            return true;
        }
        if (kind == Kind.COMPOSITE) {
            for (int i = 0; i < blockCount; i++) {
                if (blockInitializations[i].hasDiffuseComponent()) {
                    return true;
                }
            }
            return false;
        }
        if (initialDiffuseStateCov == null) {
            return false;
        }
        for (double value : initialDiffuseStateCov) {
            if (value != 0.0) {
                return true;
            }
        }
        return false;
    }

    public int dimension() {
        return dimension;
    }

    public double[] initialState() {
        return initialState;
    }

    public double[] initialStateCov() {
        return initialStateCov;
    }

    public double[] initialDiffuseStateCov() {
        return initialDiffuseStateCov;
    }

    public double approximateDiffuseVariance() {
        return approximateDiffuseVariance;
    }

    public StateInitialization resolve(StateSpaceModel model) {
        return resolve(model, 0, null);
    }

    public StateInitialization resolve(StateSpaceModel model, int t) {
        return resolve(model, t, null);
    }

    public boolean mayResolveStationary() {
        if (kind == Kind.STATIONARY) {
            return initialState == null && initialStateCov == null;
        }
        if (kind == Kind.COMPOSITE) {
            for (int i = 0; i < blockCount; i++) {
                if (blockInitializations[i].mayResolveStationary()) {
                    return true;
                }
            }
        }
        return false;
    }

    private StateInitialization resolve(StateSpaceModel model,
                                        int t,
                                        StationaryWorkspace stationaryWorkspace) {
        Objects.requireNonNull(model, "model");
        validateTimeIndex(model.observationCount(), t);
        if (kind == Kind.RESOLVED) {
            validateResolvedAgainstModel(model);
            return this;
        }
        StationaryWorkspace resolvedWorkspace = mayResolveStationary()
            ? prepareStationaryWorkspace(model, stationaryWorkspace)
            : stationaryWorkspace;
        if (kind == Kind.COMPOSITE) {
            return resolveComposite(model, t, resolvedWorkspace);
        }
        return resolveLeaf(model, t, resolvedWorkspace);
    }

    public boolean mayResolveDiffuse() {
        if (kind == Kind.DIFFUSE) {
            return true;
        }
        if (kind == Kind.COMPOSITE) {
            for (int i = 0; i < blockCount; i++) {
                if (blockInitializations[i].mayResolveDiffuse()) {
                    return true;
                }
            }
            return false;
        }
        return hasDiffuseComponent();
    }

    public boolean resolveInto(StateSpaceModel model,
                               int t,
                               double[] state,
                               int stateOffset,
                               double[] covariance,
                               int covarianceOffset,
                               double[] diffuseCovariance,
                               int diffuseCovarianceOffset) {
        return resolveInto(model, t, (StationaryWorkspace) null,
            state, stateOffset,
            covariance, covarianceOffset,
            diffuseCovariance, diffuseCovarianceOffset);
    }

    public boolean resolveInto(StateSpaceModel model,
                               int t,
                               StationaryWorkspace stationaryWorkspace,
                               double[] state,
                               int stateOffset,
                               double[] covariance,
                               int covarianceOffset,
                               double[] diffuseCovariance,
                               int diffuseCovarianceOffset) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(covariance, "covariance");
        validateTimeIndex(model.observationCount(), t);

        int blockDimension = requireCompatibleDimension(model.stateCount());
        int scalarWidth = scalarWidth(model);
        int stateLength = scalarWidth * blockDimension;
        int covarianceLength = scalarWidth * blockDimension * blockDimension;
        validateTargetCapacity("state", state, stateOffset, stateLength);
        validateTargetCapacity("covariance", covariance, covarianceOffset, covarianceLength);
        if (diffuseCovariance != null) {
            validateTargetCapacity("diffuseCovariance", diffuseCovariance, diffuseCovarianceOffset, covarianceLength);
        }

        clearSlice(state, stateOffset, stateLength);
        clearSlice(covariance, covarianceOffset, covarianceLength);
        if (diffuseCovariance != null) {
            clearSlice(diffuseCovariance, diffuseCovarianceOffset, covarianceLength);
        }

        StationaryWorkspace resolvedWorkspace = prepareStationaryWorkspace(model, stationaryWorkspace);
        return resolveIntoTargets(model, t, resolvedWorkspace,
            state, stateOffset,
            covariance, covarianceOffset,
            diffuseCovariance, diffuseCovarianceOffset,
            blockDimension, blockDimension, scalarWidth);
    }

    private static StateInitialization solveResolvedStationary(StateSpaceModel stateSpace,
                                                               int t,
                                                               StationaryWorkspace stationaryWorkspace) {
        int scalarWidth = scalarWidth(stateSpace);
        int stateLength = scalarWidth * stateSpace.stateCount();
        int covarianceLength = scalarWidth * stateSpace.stateCount() * stateSpace.stateCount();
        double[] stationaryMean = new double[stateLength];
        double[] stationaryCov = new double[covarianceLength];
        if (!solveResolvedStationaryInto(stateSpace, t, stationaryWorkspace,
                stationaryMean, 0, stationaryCov, 0, stateSpace.stateCount())) {
            return null;
        }
        return resolved(stateSpace.stateCount(), stationaryMean, stationaryCov, null);
    }

    private static boolean solveResolvedStationaryInto(StateSpaceModel stateSpace,
                                                       int t,
                                                       StationaryWorkspace stationaryWorkspace,
                                                       double[] state,
                                                       int stateOffset,
                                                       double[] covariance,
                                                       int covarianceOffset,
                                                       int covarianceDimension) {
        Objects.requireNonNull(stationaryWorkspace, "stationaryWorkspace");
        return stateSpace.complex()
            ? ComplexStationarySolver.solveCov(stateSpace, t,
                stationaryWorkspace.backing, stationaryWorkspace.pivots,
                covariance, covarianceOffset, covarianceDimension)
                && ComplexStationarySolver.solveMean(stateSpace, t,
                stationaryWorkspace.backing, stationaryWorkspace.pivots,
                state, stateOffset)
            : RealStationarySolver.solveCov(stateSpace, t,
                stationaryWorkspace.backing, stationaryWorkspace.pivots,
                covariance, covarianceOffset, covarianceDimension)
                && RealStationarySolver.solveMean(stateSpace, t,
                stationaryWorkspace.backing, stationaryWorkspace.pivots,
                state, stateOffset);
    }

    private static void validateTimeIndex(int nobs, int t) {
        if (t < 0 || t >= nobs) {
            throw new IndexOutOfBoundsException("t must be in [0, " + (nobs - 1) + "] but was " + t);
        }
    }

    private StateInitialization resolveComposite(StateSpaceModel model,
                                                 int t,
                                                 StationaryWorkspace stationaryWorkspace) {
        int blockDimension = requireCompatibleDimension(model.stateCount());
        validateCompositeCoverage(blockDimension);

        int scalarWidth = scalarWidth(model);
        double[] state = new double[scalarWidth * blockDimension];
        double[] covariance = new double[scalarWidth * blockDimension * blockDimension];
        double[] diffuseCovariance = mayResolveDiffuse() ? new double[covariance.length] : null;
        boolean diffuse = resolveIntoTargets(model, t, stationaryWorkspace,
            state, 0,
            covariance, 0,
            diffuseCovariance, 0,
            blockDimension, blockDimension, scalarWidth);
        return resolved(blockDimension, state, covariance, diffuse ? diffuseCovariance : null);
    }

    private StateInitialization resolveLeaf(StateSpaceModel model,
                                            int t,
                                            StationaryWorkspace stationaryWorkspace) {
        int blockDimension = requireCompatibleDimension(model.stateCount());
        int scalarWidth = scalarWidth(model);
        int stateLength = scalarWidth * blockDimension;
        int covarianceLength = scalarWidth * blockDimension * blockDimension;
        double[] state = new double[stateLength];
        double[] covariance = new double[covarianceLength];
        double[] diffuseCovariance = null;

        switch (kind) {
            case KNOWN:
                copyKnownState(state, stateLength);
                copyKnownCovariance(covariance, covarianceLength);
                return resolved(blockDimension, state, covariance, null);
            case DIFFUSE:
                diffuseCovariance = new double[covarianceLength];
                fillDiagonal(diffuseCovariance, blockDimension, scalarWidth, 1.0);
                return resolved(blockDimension, state, covariance, diffuseCovariance);
            case APPROXIMATE_DIFFUSE:
                copyStateIfPresent(state, stateLength);
                fillDiagonal(covariance, blockDimension, scalarWidth, approximateDiffuseVariance);
                return resolved(blockDimension, state, covariance, null);
            case STATIONARY:
                if (initialState != null || initialStateCov != null) {
                    copyKnownState(state, stateLength);
                    copyKnownCovariance(covariance, covarianceLength);
                    return resolved(blockDimension, state, covariance, null);
                }
                StateInitialization stationary = solveResolvedStationary(model, t, stationaryWorkspace);
                if (stationary == null) {
                    throw new IllegalArgumentException("State block is not discrete-stationary at t=" + t);
                }
                return stationary;
            default:
                throw new IllegalStateException("Unsupported initialization kind: " + kind);
        }
    }

    private boolean resolveIntoTargets(StateSpaceModel model,
                                       int t,
                                       StationaryWorkspace stationaryWorkspace,
                                       double[] state,
                                       int stateOffset,
                                       double[] covariance,
                                       int covarianceOffset,
                                       double[] diffuseCovariance,
                                       int diffuseCovarianceOffset,
                                       int blockDimension,
                                       int targetDimension,
                                       int scalarWidth) {
        if (kind == Kind.RESOLVED) {
            validateResolvedAgainstModel(model);
            copySlice(initialState, state, stateOffset);
            copySquareSlice(initialStateCov, blockDimension, covariance, covarianceOffset, targetDimension, scalarWidth);
            if (initialDiffuseStateCov != null && diffuseCovariance != null) {
                copySquareSlice(initialDiffuseStateCov, blockDimension, diffuseCovariance, diffuseCovarianceOffset, targetDimension, scalarWidth);
            }
            return hasDiffuseComponent();
        }
        if (kind == Kind.COMPOSITE) {
            validateCompositeCoverage(blockDimension);
            boolean hasDiffuse = false;
            for (int i = 0; i < blockCount; i++) {
                int start = blockStarts[i];
                int end = blockEnds[i];
                StateSpaceModel blockModel = buildStateBlockModel(model, t, start, end - start);
                int blockDimensionSize = end - start;
                int blockStateOffset = stateOffset + start * scalarWidth;
                int blockCovarianceOffset = covarianceOffset + start * scalarWidth * targetDimension + start * scalarWidth;
                int blockDiffuseOffset = diffuseCovarianceOffset + start * scalarWidth * targetDimension + start * scalarWidth;
                hasDiffuse |= blockInitializations[i].resolveIntoTargets(blockModel, 0,
                    stationaryWorkspace,
                    state, blockStateOffset,
                    covariance, blockCovarianceOffset,
                    diffuseCovariance, blockDiffuseOffset,
                    blockDimensionSize, targetDimension, scalarWidth);
            }
            return hasDiffuse;
        }

        switch (kind) {
            case KNOWN:
                copySlice(initialState, state, stateOffset);
                copySquareSlice(initialStateCov, blockDimension, covariance, covarianceOffset, targetDimension, scalarWidth);
                return false;
            case DIFFUSE:
                if (diffuseCovariance != null) {
                    fillDiagonal(diffuseCovariance, diffuseCovarianceOffset, targetDimension, blockDimension, scalarWidth, 1.0);
                }
                return true;
            case APPROXIMATE_DIFFUSE:
                copySlice(initialState, state, stateOffset);
                fillDiagonal(covariance, covarianceOffset, targetDimension, blockDimension, scalarWidth, approximateDiffuseVariance);
                return false;
            case STATIONARY:
                if (initialState != null || initialStateCov != null) {
                    copySlice(initialState, state, stateOffset);
                    copySquareSlice(initialStateCov, blockDimension, covariance, covarianceOffset, targetDimension, scalarWidth);
                    return false;
                }
                if (!solveResolvedStationaryInto(model, t, stationaryWorkspace,
                        state, stateOffset, covariance, covarianceOffset, targetDimension)) {
                    throw new IllegalArgumentException("State block is not discrete-stationary at t=" + t);
                }
                return false;
            default:
                throw new IllegalStateException("Unsupported initialization kind: " + kind);
        }
    }

    private void validateResolvedAgainstModel(StateSpaceModel model) {
        int blockDimension = requireCompatibleDimension(model.stateCount());
        int scalarWidth = scalarWidth(model);
        int expectedStateLength = scalarWidth * blockDimension;
        int expectedCovarianceLength = scalarWidth * blockDimension * blockDimension;
        if (initialState == null || initialState.length != expectedStateLength) {
            throw new IllegalArgumentException("resolved initialState length mismatch for model");
        }
        if (initialStateCov == null || initialStateCov.length != expectedCovarianceLength) {
            throw new IllegalArgumentException("resolved initialStateCov length mismatch for model");
        }
        if (initialDiffuseStateCov != null && initialDiffuseStateCov.length != expectedCovarianceLength) {
            throw new IllegalArgumentException("resolved initialDiffuseStateCov length mismatch for model");
        }
    }

    private static StateInitialization resolved(int dimension,
                                                double[] initialState,
                                                double[] initialStateCov,
                                                double[] initialDiffuseStateCov) {
        return new StateInitialization(Kind.RESOLVED, dimension,
            initialState, initialStateCov, initialDiffuseStateCov, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    private static StateInitialization solveStationary(StateSpaceModel stateSpace, int t) {
        StationaryWorkspace stationaryWorkspace = prepareStationaryWorkspace(stateSpace, null);
        int scalarWidth = scalarWidth(stateSpace);
        int stateLength = scalarWidth * stateSpace.stateCount();
        int covarianceLength = scalarWidth * stateSpace.stateCount() * stateSpace.stateCount();
        double[] stationaryMean = new double[stateLength];
        double[] stationaryCov = new double[covarianceLength];
        if (!solveResolvedStationaryInto(stateSpace, t, stationaryWorkspace,
                stationaryMean, 0, stationaryCov, 0, stateSpace.stateCount())) {
            return null;
        }
        return new StateInitialization(Kind.STATIONARY, stateSpace.stateCount(),
            stationaryMean, stationaryCov, null, DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE);
    }

    private void copyKnownState(double[] target, int expectedLength) {
        if (initialState == null) {
            return;
        }
        validateLength("initialState", initialState, expectedLength);
        System.arraycopy(initialState, 0, target, 0, expectedLength);
    }

    private void copyKnownCovariance(double[] target, int expectedLength) {
        if (initialStateCov == null) {
            return;
        }
        validateLength("initialStateCov", initialStateCov, expectedLength);
        System.arraycopy(initialStateCov, 0, target, 0, expectedLength);
    }

    private void copyStateIfPresent(double[] target, int expectedLength) {
        if (initialState == null) {
            return;
        }
        validateLength("initialState", initialState, expectedLength);
        System.arraycopy(initialState, 0, target, 0, expectedLength);
    }

    private static void copySlice(double[] source, double[] target, int targetOffset) {
        if (source == null || source.length == 0) {
            return;
        }
        System.arraycopy(source, 0, target, targetOffset, source.length);
    }

    private static void copySquareSlice(double[] source,
                                        int sourceDimension,
                                        double[] target,
                                        int targetOffset,
                                        int targetDimension,
                                        int scalarWidth) {
        if (source == null || source.length == 0) {
            return;
        }
        int sourceRowWidth = scalarWidth * sourceDimension;
        int targetRowWidth = scalarWidth * targetDimension;
        for (int row = 0; row < sourceDimension; row++) {
            System.arraycopy(source, row * sourceRowWidth, target, targetOffset + row * targetRowWidth, sourceRowWidth);
        }
    }

    private static void clearSlice(double[] values, int offset, int length) {
        Arrays.fill(values, offset, offset + length, 0.0);
    }

    private static void validateTargetCapacity(String name, double[] values, int offset, int requiredLength) {
        if (offset < 0 || requiredLength < 0 || offset + requiredLength > values.length) {
            throw new IllegalArgumentException(name + " capacity " + values.length + " is too small for offset "
                + offset + " and required length " + requiredLength);
        }
    }

    private void validateCompositeCoverage(int blockDimension) {
        if (blockCount == 0) {
            throw new IllegalArgumentException("Composite initialization must cover all state blocks before resolve");
        }
        int cursor = 0;
        for (int i = 0; i < blockCount; i++) {
            if (blockStarts[i] != cursor) {
                throw new IllegalArgumentException("Composite initialization leaves state gap at index " + cursor);
            }
            cursor = blockEnds[i];
        }
        if (cursor != blockDimension) {
            throw new IllegalArgumentException("Composite initialization leaves trailing state gap at index " + cursor);
        }
    }

    private int requireCompatibleDimension(int modelDimension) {
        if (dimension != UNSPECIFIED_DIMENSION && dimension != modelDimension) {
            throw new IllegalArgumentException("Initialization dimension " + dimension + " does not match model state count " + modelDimension);
        }
        return modelDimension;
    }

    private static int scalarWidth(StateSpaceModel model) {
        return model.complex() ? 2 : 1;
    }

    private static void fillDiagonal(double[] matrix, int offset, int dimension, int scalarWidth, double value) {
        fillDiagonal(matrix, offset, dimension, dimension, scalarWidth, value);
    }

    private static void fillDiagonal(double[] matrix, int offset, int targetDimension, int blockDimension, int scalarWidth, double value) {
        for (int i = 0; i < blockDimension; i++) {
            int diagonal = offset + i * scalarWidth * targetDimension + i * scalarWidth;
            matrix[diagonal] = value;
            if (scalarWidth == 2) {
                matrix[diagonal + 1] = 0.0;
            }
        }
    }

    private static void fillDiagonal(double[] matrix, int dimension, int scalarWidth, double value) {
        fillDiagonal(matrix, 0, dimension, scalarWidth, value);
    }

    private static void validateKnownInputs(double[] initialState, double[] initialStateCov) {
        if (initialState == null && initialStateCov == null) {
            throw new IllegalArgumentException("known initialization requires an initial state or covariance");
        }
    }

    private static void validateApproximateDiffuseVariance(double approximateDiffuseVariance) {
        if (!(approximateDiffuseVariance > 0.0) || !Double.isFinite(approximateDiffuseVariance)) {
            throw new IllegalArgumentException("approximateDiffuseVariance must be finite and positive");
        }
    }

    private static void validateCompositeDimension(int dimension) {
        if (dimension < 0) {
            throw new IllegalArgumentException("dimension must be non-negative");
        }
    }

    private static void validateLength(String name, double[] values, int expectedLength) {
        if (values.length != expectedLength) {
            throw new IllegalArgumentException(name + " length " + values.length + " does not match expected " + expectedLength);
        }
    }

    private void requireComposite() {
        if (kind != Kind.COMPOSITE) {
            throw new IllegalStateException("set/unset/clear are only valid for composite initializations");
        }
    }

    private static void validateRange(int startInclusive, int endExclusive, int dimension) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > dimension) {
            throw new IllegalArgumentException("range [" + startInclusive + ", " + endExclusive + ") is out of bounds for dimension " + dimension);
        }
    }

    private static void validateBlockDimension(StateInitialization initialization, int expectedDimension) {
        if (initialization.dimension != UNSPECIFIED_DIMENSION && initialization.dimension != expectedDimension) {
            throw new IllegalArgumentException("block dimension " + initialization.dimension
                + " does not match target range dimension " + expectedDimension);
        }
    }

    private void insertBlock(int startInclusive, int endExclusive, StateInitialization initialization) {
        int insertionPoint = 0;
        while (insertionPoint < blockCount && blockStarts[insertionPoint] < startInclusive) {
            insertionPoint++;
        }
        if (insertionPoint > 0 && blockEnds[insertionPoint - 1] > startInclusive) {
            throw new IllegalArgumentException("Initialization blocks must not overlap");
        }
        if (insertionPoint < blockCount && blockStarts[insertionPoint] < endExclusive) {
            throw new IllegalArgumentException("Initialization blocks must not overlap");
        }

        ensureBlockCapacity(blockCount + 1);
        if (insertionPoint < blockCount) {
            System.arraycopy(blockStarts, insertionPoint, blockStarts, insertionPoint + 1, blockCount - insertionPoint);
            System.arraycopy(blockEnds, insertionPoint, blockEnds, insertionPoint + 1, blockCount - insertionPoint);
            System.arraycopy(blockInitializations, insertionPoint, blockInitializations, insertionPoint + 1, blockCount - insertionPoint);
        }
        blockStarts[insertionPoint] = startInclusive;
        blockEnds[insertionPoint] = endExclusive;
        blockInitializations[insertionPoint] = initialization;
        blockCount++;
    }

    private int findExactBlock(int startInclusive, int endExclusive) {
        for (int i = 0; i < blockCount; i++) {
            if (blockStarts[i] == startInclusive && blockEnds[i] == endExclusive) {
                return i;
            }
        }
        return -1;
    }

    private void removeBlock(int index) {
        if (index + 1 < blockCount) {
            System.arraycopy(blockStarts, index + 1, blockStarts, index, blockCount - index - 1);
            System.arraycopy(blockEnds, index + 1, blockEnds, index, blockCount - index - 1);
            System.arraycopy(blockInitializations, index + 1, blockInitializations, index, blockCount - index - 1);
        }
        blockCount--;
    }

    private void ensureBlockCapacity(int capacity) {
        if (blockStarts == null || blockStarts.length < capacity) {
            int nextCapacity = Math.max(capacity, blockStarts == null ? 4 : blockStarts.length * 2);
            blockStarts = blockStarts == null ? new int[nextCapacity] : Arrays.copyOf(blockStarts, nextCapacity);
            blockEnds = blockEnds == null ? new int[nextCapacity] : Arrays.copyOf(blockEnds, nextCapacity);
            blockInitializations = blockInitializations == null
                ? new StateInitialization[nextCapacity]
                : Arrays.copyOf(blockInitializations, nextCapacity);
        }
    }

    private static StateSpaceModel buildStateBlockModel(StateSpaceModel model,
                                                        int t,
                                                        int startState,
                                                        int blockDimension) {
        int scalarWidth = scalarWidth(model);
        int kPosdef = model.stateDisturbanceCount();
        double[] transition = copySquareBlock(model.transitionData(), model.transitionOffset(t),
            model.transitionLeadingDimension(), startState, blockDimension, scalarWidth);
        double[] stateIntercept = copyVectorBlock(model.stateInterceptData(), model.stateInterceptOffset(t),
            startState, blockDimension, scalarWidth);
        double[] selection = copyRectangularBlock(model.selectionData(), model.selectionOffset(t),
            model.selectionLeadingDimension(), startState, blockDimension, kPosdef, scalarWidth);
        double[] stateCovariance = copySquareBlock(model.stateCovarianceData(), model.stateCovarianceOffset(t),
            model.stateCovarianceLeadingDimension(), 0, kPosdef, scalarWidth);

        StateSpaceModel.Builder builder = model.complex()
            ? StateSpaceModel.complexBuilder(1, blockDimension, kPosdef, 1)
            : StateSpaceModel.builder(1, blockDimension, kPosdef, 1);
        return builder
            .design(new double[scalarWidth * blockDimension], false)
            .obsIntercept(new double[scalarWidth], false)
            .obsCov(new double[scalarWidth], false)
            .transition(transition, false)
            .stateIntercept(stateIntercept, false)
            .selection(selection, false)
            .stateCovariance(stateCovariance, false)
            .endog(new double[scalarWidth])
            .allObserved()
            .build();
    }

    private static double[] copyVectorBlock(double[] source,
                                            int offset,
                                            int startState,
                                            int blockDimension,
                                            int scalarWidth) {
        double[] values = new double[scalarWidth * blockDimension];
        if (source == null) {
            return values;
        }
        System.arraycopy(source, offset + startState * scalarWidth, values, 0, scalarWidth * blockDimension);
        return values;
    }

    private static double[] copyRectangularBlock(double[] source,
                                                 int offset,
                                                 int leadingDimension,
                                                 int startRow,
                                                 int rows,
                                                 int cols,
                                                 int scalarWidth) {
        double[] values = new double[rows * scalarWidth * cols];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(source,
                offset + (startRow + row) * scalarWidth * leadingDimension,
                values,
                row * scalarWidth * cols,
                scalarWidth * cols);
        }
        return values;
    }

    private static double[] copySquareBlock(double[] source,
                                            int offset,
                                            int leadingDimension,
                                            int startState,
                                            int blockDimension,
                                            int scalarWidth) {
        double[] values = new double[blockDimension * scalarWidth * blockDimension];
        for (int row = 0; row < blockDimension; row++) {
            int sourceRow = offset + (startState + row) * scalarWidth * leadingDimension + startState * scalarWidth;
            int targetRow = row * scalarWidth * blockDimension;
            System.arraycopy(source, sourceRow, values, targetRow, scalarWidth * blockDimension);
        }
        return values;
    }

    private static StationaryWorkspace prepareStationaryWorkspace(StateSpaceModel model,
                                                                 StationaryWorkspace stationaryWorkspace) {
        if (stationaryWorkspace == null) {
            stationaryWorkspace = new StationaryWorkspace();
        }
        stationaryWorkspace.ensureCapacity(model);
        return stationaryWorkspace;
    }

    public static int requiredStationaryBackingLength(StateSpaceModel model) {
        return model.complex()
            ? ComplexStationarySolver.requiredDoubleCount(model.stateCount(), model.stateDisturbanceCount())
            : RealStationarySolver.requiredDoubleCount(model.stateCount(), model.stateDisturbanceCount());
    }

    public static int requiredStationaryPivotLength(StateSpaceModel model) {
        return model.complex()
            ? ComplexStationarySolver.requiredPivotCount(model.stateCount())
            : RealStationarySolver.requiredPivotCount(model.stateCount());
    }
}