package com.curioloop.yum4j.kalman.model;

import java.util.Objects;

/**
 * Unified dense state-space model with BLAS-friendly flat storage.
 */
public class StateSpaceModel {

    private static final int REAL_SCALAR_WIDTH = 1;
    private static final int COMPLEX_SCALAR_WIDTH = 2;

    public final int kEndog;
    public final int kStates;
    public final int kPosdef;
    public final int nobs;

    private final int scalarWidth;

    private final int transitionStride;
    private final int stateInterceptStride;
    private final int selectionStride;
    private final int stateCovStride;
    private final int designStride;
    private final int obsInterceptStride;
    private final int obsCovStride;
    private final int endogStride;
    private final int endogBase;

    public double[] design;
    public double[] obsIntercept;
    public double[] obsCov;
    public double[] transition;
    public double[] stateIntercept;
    public double[] selection;
    public double[] stateCov;
    public double[] endog;
    public boolean[] missing;
    public int[] nmissing;

    protected StateSpaceModel() {
        this.kEndog = 0;
        this.kStates = 0;
        this.kPosdef = 0;
        this.nobs = 0;
        this.scalarWidth = REAL_SCALAR_WIDTH;
        this.transitionStride = 0;
        this.stateInterceptStride = 0;
        this.selectionStride = 0;
        this.stateCovStride = 0;
        this.designStride = 0;
        this.obsInterceptStride = 0;
        this.obsCovStride = 0;
        this.endogStride = 0;
        this.endogBase = 0;
    }

    public StateSpaceModel(int kEndog, int kStates, int kPosdef, int nobs) {
        this(kEndog, kStates, kPosdef, nobs, REAL_SCALAR_WIDTH,
            new double[designBlockSize(kEndog, kStates, REAL_SCALAR_WIDTH) * nobs],
            designBlockSize(kEndog, kStates, REAL_SCALAR_WIDTH),
            new double[obsVectorBlockSize(kEndog, REAL_SCALAR_WIDTH) * nobs],
            obsVectorBlockSize(kEndog, REAL_SCALAR_WIDTH),
            new double[obsCovBlockSize(kEndog, REAL_SCALAR_WIDTH) * nobs],
            obsCovBlockSize(kEndog, REAL_SCALAR_WIDTH),
            new double[transitionBlockSize(kStates, REAL_SCALAR_WIDTH) * nobs],
            transitionBlockSize(kStates, REAL_SCALAR_WIDTH),
            new double[stateVectorBlockSize(kStates, REAL_SCALAR_WIDTH) * nobs],
            stateVectorBlockSize(kStates, REAL_SCALAR_WIDTH),
            new double[selectionBlockSize(kStates, kPosdef, REAL_SCALAR_WIDTH) * nobs],
            selectionBlockSize(kStates, kPosdef, REAL_SCALAR_WIDTH),
            new double[stateCovBlockSize(kPosdef, REAL_SCALAR_WIDTH) * nobs],
            stateCovBlockSize(kPosdef, REAL_SCALAR_WIDTH),
            new double[obsVectorBlockSize(kEndog, REAL_SCALAR_WIDTH) * nobs],
            0,
            obsVectorBlockSize(kEndog, REAL_SCALAR_WIDTH),
            new boolean[kEndog * nobs],
            new int[nobs]);
    }

    private StateSpaceModel(int kEndog,
                            int kStates,
                            int kPosdef,
                            int nobs,
                            int scalarWidth,
                            double[] design,
                            int designStride,
                            double[] obsIntercept,
                            int obsInterceptStride,
                            double[] obsCov,
                            int obsCovStride,
                            double[] transition,
                            int transitionStride,
                            double[] stateIntercept,
                            int stateInterceptStride,
                            double[] selection,
                            int selectionStride,
                            double[] stateCov,
                            int stateCovStride,
                            double[] endog,
                            int endogBase,
                            int endogStride,
                            boolean[] missing,
                            int[] nmissing) {
        validateDimensions(kEndog, kStates, kPosdef, nobs);
        validateScalarWidth(scalarWidth);
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        this.scalarWidth = scalarWidth;
        this.design = validateComponent("design", design,
            designBlockSize(kEndog, kStates, scalarWidth), nobs, designStride, 0);
        this.designStride = designStride;
        this.obsIntercept = validateComponent("obsIntercept", obsIntercept,
            obsVectorBlockSize(kEndog, scalarWidth), nobs, obsInterceptStride, 0);
        this.obsInterceptStride = obsInterceptStride;
        this.obsCov = validateComponent("obsCov", obsCov,
            obsCovBlockSize(kEndog, scalarWidth), nobs, obsCovStride, 0);
        this.obsCovStride = obsCovStride;
        this.transition = validateComponent("transition", transition,
            transitionBlockSize(kStates, scalarWidth), nobs, transitionStride, 0);
        this.transitionStride = transitionStride;
        this.stateIntercept = validateComponent("stateIntercept", stateIntercept,
            stateVectorBlockSize(kStates, scalarWidth), nobs, stateInterceptStride, 0);
        this.stateInterceptStride = stateInterceptStride;
        this.selection = validateComponent("selection", selection,
            selectionBlockSize(kStates, kPosdef, scalarWidth), nobs, selectionStride, 0);
        this.selectionStride = selectionStride;
        this.stateCov = validateComponent("stateCov", stateCov,
            stateCovBlockSize(kPosdef, scalarWidth), nobs, stateCovStride, 0);
        this.stateCovStride = stateCovStride;
        this.endog = validateComponent("endog", endog,
            obsVectorBlockSize(kEndog, scalarWidth), nobs, endogStride, endogBase);
        this.endogBase = endogBase;
        this.endogStride = endogStride;
        validateMissingArrays(missing, nmissing, kEndog, nobs);
        this.missing = missing;
        this.nmissing = nmissing == null ? computeMissingCounts(missing, kEndog, nobs) : nmissing;
    }

    public static StateSpaceModel complex(int kEndog, int kStates, int kPosdef, int nobs) {
        return new StateSpaceModel(kEndog, kStates, kPosdef, nobs, COMPLEX_SCALAR_WIDTH,
            new double[designBlockSize(kEndog, kStates, COMPLEX_SCALAR_WIDTH) * nobs],
            designBlockSize(kEndog, kStates, COMPLEX_SCALAR_WIDTH),
            new double[obsVectorBlockSize(kEndog, COMPLEX_SCALAR_WIDTH) * nobs],
            obsVectorBlockSize(kEndog, COMPLEX_SCALAR_WIDTH),
            new double[obsCovBlockSize(kEndog, COMPLEX_SCALAR_WIDTH) * nobs],
            obsCovBlockSize(kEndog, COMPLEX_SCALAR_WIDTH),
            new double[transitionBlockSize(kStates, COMPLEX_SCALAR_WIDTH) * nobs],
            transitionBlockSize(kStates, COMPLEX_SCALAR_WIDTH),
            new double[stateVectorBlockSize(kStates, COMPLEX_SCALAR_WIDTH) * nobs],
            stateVectorBlockSize(kStates, COMPLEX_SCALAR_WIDTH),
            new double[selectionBlockSize(kStates, kPosdef, COMPLEX_SCALAR_WIDTH) * nobs],
            selectionBlockSize(kStates, kPosdef, COMPLEX_SCALAR_WIDTH),
            new double[stateCovBlockSize(kPosdef, COMPLEX_SCALAR_WIDTH) * nobs],
            stateCovBlockSize(kPosdef, COMPLEX_SCALAR_WIDTH),
            new double[obsVectorBlockSize(kEndog, COMPLEX_SCALAR_WIDTH) * nobs],
            0,
            obsVectorBlockSize(kEndog, COMPLEX_SCALAR_WIDTH),
            new boolean[kEndog * nobs],
            new int[nobs]);
    }

    public static Builder builder(int kEndog, int kStates, int kPosdef, int nobs) {
        return new Builder(kEndog, kStates, kPosdef, nobs, REAL_SCALAR_WIDTH);
    }

    public static Builder complexBuilder(int kEndog, int kStates, int kPosdef, int nobs) {
        return new Builder(kEndog, kStates, kPosdef, nobs, COMPLEX_SCALAR_WIDTH);
    }

    public static StateSpaceModel copyOf(StateSpaceModel source) {
        Objects.requireNonNull(source, "source");
        int scalarWidth = scalarWidth(source.complex());
        boolean hasMissing = hasMissingObservations(source);
        return new StateSpaceModel(source.observationDimension(),
            source.stateCount(),
            source.stateDisturbanceCount(),
            source.observationCount(),
            scalarWidth,
            copyDesign(source, scalarWidth),
            designBlockSize(source.observationDimension(), source.stateCount(), scalarWidth),
            copyObservationVector(source.obsInterceptData(), source::obsInterceptOffset,
                source.observationDimension(), source.observationCount(), scalarWidth),
            obsVectorBlockSize(source.observationDimension(), scalarWidth),
            copySquareBlocks(source.obsCovData(), source::obsCovOffset,
                source.obsCovLeadingDimension(), source.observationDimension(), source.observationCount(), scalarWidth),
            obsCovBlockSize(source.observationDimension(), scalarWidth),
            copySquareBlocks(source.transitionData(), source::transitionOffset,
                source.transitionLeadingDimension(), source.stateCount(), source.observationCount(), scalarWidth),
            transitionBlockSize(source.stateCount(), scalarWidth),
            copyStateVector(source.stateInterceptData(), source::stateInterceptOffset,
                source.stateCount(), source.observationCount(), scalarWidth),
            stateVectorBlockSize(source.stateCount(), scalarWidth),
            copyRectangularBlocks(source.selectionData(), source::selectionOffset,
                source.selectionLeadingDimension(), source.stateCount(), source.stateDisturbanceCount(), source.observationCount(), scalarWidth),
            selectionBlockSize(source.stateCount(), source.stateDisturbanceCount(), scalarWidth),
            copySquareBlocks(source.stateCovarianceData(), source::stateCovarianceOffset,
                source.stateCovarianceLeadingDimension(), source.stateDisturbanceCount(), source.observationCount(), scalarWidth),
            stateCovBlockSize(source.stateDisturbanceCount(), scalarWidth),
            copyObservationVector(source.endogData(), source::endogOffset,
                source.observationDimension(), source.observationCount(), scalarWidth),
            0,
            obsVectorBlockSize(source.observationDimension(), scalarWidth),
            hasMissing ? copyMissingFlags(source) : null,
            hasMissing ? copyMissingCounts(source) : null);
    }

    public int stateCount() {
        return kStates;
    }

    public int stateDisturbanceCount() {
        return kPosdef;
    }

    public int observationDimension() {
        return kEndog;
    }

    public int observationCount() {
        return nobs;
    }

    /**
     * Whether returned flat arrays use interleaved complex storage.
     */
    public boolean complex() {
        return scalarWidth == COMPLEX_SCALAR_WIDTH;
    }

    public boolean isTimeInvariant() {
        return designStride == 0
            && obsInterceptStride == 0
            && obsCovStride == 0
            && transitionStride == 0
            && stateInterceptStride == 0
            && selectionStride == 0
            && stateCovStride == 0;
    }

    public double[] transitionData() {
        return transition;
    }

    public int transitionOffset(int t) {
        return t * transitionStride;
    }

    public int transitionLeadingDimension() {
        return kStates;
    }

    public double[] stateInterceptData() {
        return stateIntercept;
    }

    public int stateInterceptOffset(int t) {
        return t * stateInterceptStride;
    }

    public double[] selectionData() {
        return selection;
    }

    public int selectionOffset(int t) {
        return t * selectionStride;
    }

    public int selectionLeadingDimension() {
        return kPosdef;
    }

    public double[] stateCovarianceData() {
        return stateCov;
    }

    public int stateCovarianceOffset(int t) {
        return t * stateCovStride;
    }

    public int stateCovOffset(int t) {
        return stateCovarianceOffset(t);
    }

    public int stateCovarianceLeadingDimension() {
        return kPosdef;
    }

    public double[] designData() {
        return design;
    }

    public int designOffset(int t) {
        return t * designStride;
    }

    public int designLeadingDimension() {
        return kStates;
    }

    public double[] obsInterceptData() {
        return obsIntercept;
    }

    public int obsInterceptOffset(int t) {
        return t * obsInterceptStride;
    }

    public double[] obsCovData() {
        return obsCov;
    }

    public int obsCovOffset(int t) {
        return t * obsCovStride;
    }

    public int obsCovLeadingDimension() {
        return kEndog;
    }

    public double[] endogData() {
        return endog;
    }

    public int endogOffset(int t) {
        return endogBase + t * endogStride;
    }

    public boolean isMissing(int obsIndex, int t) {
        return missing != null && missing[missingOffset(t) + obsIndex];
    }

    public int missingCount(int t) {
        return nmissing == null ? 0 : nmissing[t];
    }

    public void setDesign(double[] values, int t) {
        System.arraycopy(values, 0, design, designOffset(t), kEndog * scalarWidth * kStates);
    }

    public double[] getDesign() {
        return design;
    }

    public double[] getDesign(int t) {
        return design;
    }

    public void setObsIntercept(double[] values, int t) {
        System.arraycopy(values, 0, obsIntercept, obsInterceptOffset(t), scalarWidth * kEndog);
    }

    public double[] getObsIntercept() {
        return obsIntercept;
    }

    public double[] getObsIntercept(int t) {
        return obsIntercept;
    }

    public void setObsCov(double[] values, int t) {
        System.arraycopy(values, 0, obsCov, obsCovOffset(t), kEndog * scalarWidth * kEndog);
    }

    public double[] getObsCov() {
        return obsCov;
    }

    public double[] getObsCov(int t) {
        return obsCov;
    }

    public void setTransition(double[] values, int t) {
        System.arraycopy(values, 0, transition, transitionOffset(t), kStates * scalarWidth * kStates);
    }

    public double[] getTransition() {
        return transition;
    }

    public double[] getTransition(int t) {
        return transition;
    }

    public void setStateIntercept(double[] values, int t) {
        System.arraycopy(values, 0, stateIntercept, stateInterceptOffset(t), scalarWidth * kStates);
    }

    public double[] getStateIntercept() {
        return stateIntercept;
    }

    public double[] getStateIntercept(int t) {
        return stateIntercept;
    }

    public void setSelection(double[] values, int t) {
        System.arraycopy(values, 0, selection, selectionOffset(t), kStates * scalarWidth * kPosdef);
    }

    public double[] getSelection() {
        return selection;
    }

    public double[] getSelection(int t) {
        return selection;
    }

    public void setStateCov(double[] values, int t) {
        System.arraycopy(values, 0, stateCov, stateCovOffset(t), kPosdef * scalarWidth * kPosdef);
    }

    public double[] getStateCov() {
        return stateCov;
    }

    public double[] getStateCov(int t) {
        return stateCov;
    }

    public void setEndog(double[] values, int t) {
        System.arraycopy(values, 0, endog, endogOffset(t), scalarWidth * kEndog);
    }

    public double[] getEndog() {
        return endog;
    }

    public double[] getEndog(int t) {
        return endog;
    }

    public boolean[] getMissing() {
        return missing;
    }

    public int[] getNmissing() {
        return nmissing;
    }

    public void setMissing(boolean[] mask, int t) {
        ensureMissingArrays();
        int count = 0;
        int offset = missingOffset(t);
        for (int obs = 0; obs < kEndog; obs++) {
            boolean value = mask[obs];
            missing[offset + obs] = value;
            if (value) {
                count++;
            }
        }
        nmissing[t] = count;
    }

    public long retainedModelDoubleCount() {
        return doubleCount(design)
                + doubleCount(obsIntercept)
                + doubleCount(obsCov)
                + doubleCount(transition)
                + doubleCount(stateIntercept)
                + doubleCount(selection)
                + doubleCount(stateCov)
                + doubleCount(endog);
    }

    public long retainedTotalByteCount() {
        return retainedModelDoubleCount() * Double.BYTES + retainedMetadataByteCount();
    }

    private void ensureMissingArrays() {
        if (missing == null) {
            missing = new boolean[kEndog * nobs];
        }
        if (nmissing == null) {
            nmissing = new int[nobs];
        }
    }

    private int missingOffset(int t) {
        return t * kEndog;
    }

    private long retainedMetadataByteCount() {
        return booleanByteCount(missing) + intByteCount(nmissing);
    }

    private static void validateDimensions(int kEndog, int kStates, int kPosdef, int nobs) {
        if (kEndog < 0 || kStates < 0 || kPosdef < 0 || nobs < 0) {
            throw new IllegalArgumentException("dimensions must be non-negative");
        }
    }

    private static void validateScalarWidth(int scalarWidth) {
        if (scalarWidth != REAL_SCALAR_WIDTH && scalarWidth != COMPLEX_SCALAR_WIDTH) {
            throw new IllegalArgumentException("scalarWidth must be 1 or 2");
        }
    }

    private static double[] validateComponent(String name,
                                              double[] values,
                                              int blockSize,
                                              int nobs,
                                              int stride,
                                              int base) {
        Objects.requireNonNull(values, name);
        if (stride < 0 || base < 0) {
            throw new IllegalArgumentException(name + " stride/base must be non-negative");
        }
        int required = requiredLength(blockSize, nobs, stride, base);
        if (values.length < required) {
            throw new IllegalArgumentException(name + " length " + values.length + " is smaller than required " + required);
        }
        return values;
    }

    private static int requiredLength(int blockSize, int nobs, int stride, int base) {
        if (nobs == 0) {
            return base;
        }
        return base + (nobs - 1) * stride + blockSize;
    }

    private static void validateMissingArrays(boolean[] missing, int[] nmissing, int kEndog, int nobs) {
        if (missing == null) {
            if (nmissing != null) {
                throw new IllegalArgumentException("nmissing requires missing flags");
            }
            return;
        }
        if (missing.length < kEndog * nobs) {
            throw new IllegalArgumentException("missing length is smaller than required");
        }
        if (nmissing != null && nmissing.length < nobs) {
            throw new IllegalArgumentException("nmissing length is smaller than required");
        }
    }

    private static int[] computeMissingCounts(boolean[] missing, int kEndog, int nobs) {
        if (missing == null) {
            return null;
        }
        int[] counts = new int[nobs];
        for (int t = 0; t < nobs; t++) {
            int off = t * kEndog;
            int count = 0;
            for (int obs = 0; obs < kEndog; obs++) {
                if (missing[off + obs]) {
                    count++;
                }
            }
            counts[t] = count;
        }
        return counts;
    }

    private static int designBlockSize(int kEndog, int kStates, int scalarWidth) {
        return kEndog * scalarWidth * kStates;
    }

    private static int obsVectorBlockSize(int kEndog, int scalarWidth) {
        return scalarWidth * kEndog;
    }

    private static int obsCovBlockSize(int kEndog, int scalarWidth) {
        return kEndog * scalarWidth * kEndog;
    }

    private static int transitionBlockSize(int kStates, int scalarWidth) {
        return kStates * scalarWidth * kStates;
    }

    private static int stateVectorBlockSize(int kStates, int scalarWidth) {
        return scalarWidth * kStates;
    }

    private static int selectionBlockSize(int kStates, int kPosdef, int scalarWidth) {
        return kStates * scalarWidth * kPosdef;
    }

    private static int stateCovBlockSize(int kPosdef, int scalarWidth) {
        return kPosdef * scalarWidth * kPosdef;
    }

    private static int scalarWidth(boolean complex) {
        return complex ? COMPLEX_SCALAR_WIDTH : REAL_SCALAR_WIDTH;
    }

    private static boolean hasMissingObservations(StateSpaceModel model) {
        for (int t = 0; t < model.observationCount(); t++) {
            if (model.missingCount(t) > 0) {
                return true;
            }
        }
        return false;
    }

    private static double[] copyDesign(StateSpaceModel model, int scalarWidth) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        double[] source = model.designData();
        double[] copy = new double[kEndog * scalarWidth * kStates * nobs];
        int sourceLd = model.designLeadingDimension();
        for (int t = 0; t < nobs; t++) {
            int sourceOffset = model.designOffset(t);
            int targetOffset = t * kEndog * scalarWidth * kStates;
            for (int row = 0; row < kEndog; row++) {
                System.arraycopy(source,
                        sourceOffset + row * scalarWidth * sourceLd,
                        copy,
                        targetOffset + row * scalarWidth * kStates,
                        scalarWidth * kStates);
            }
        }
        return copy;
    }

    private static double[] copyObservationVector(double[] source,
                                                  IntOffset offsetFn,
                                                  int width,
                                                  int nobs,
                                                  int scalarWidth) {
        double[] copy = new double[scalarWidth * width * nobs];
        for (int t = 0; t < nobs; t++) {
            System.arraycopy(source, offsetFn.offset(t), copy, t * scalarWidth * width, scalarWidth * width);
        }
        return copy;
    }

    private static double[] copyStateVector(double[] source,
                                            IntOffset offsetFn,
                                            int width,
                                            int nobs,
                                            int scalarWidth) {
        return copyObservationVector(source, offsetFn, width, nobs, scalarWidth);
    }

    private static double[] copySquareBlocks(double[] source,
                                             IntOffset offsetFn,
                                             int leadingDimension,
                                             int width,
                                             int nobs,
                                             int scalarWidth) {
        double[] copy = new double[width * scalarWidth * width * nobs];
        for (int t = 0; t < nobs; t++) {
            int sourceOffset = offsetFn.offset(t);
            int targetOffset = t * width * scalarWidth * width;
            for (int row = 0; row < width; row++) {
                System.arraycopy(source,
                        sourceOffset + row * scalarWidth * leadingDimension,
                        copy,
                        targetOffset + row * scalarWidth * width,
                        scalarWidth * width);
            }
        }
        return copy;
    }

    private static double[] copyRectangularBlocks(double[] source,
                                                  IntOffset offsetFn,
                                                  int leadingDimension,
                                                  int rows,
                                                  int cols,
                                                  int nobs,
                                                  int scalarWidth) {
        double[] copy = new double[rows * scalarWidth * cols * nobs];
        for (int t = 0; t < nobs; t++) {
            int sourceOffset = offsetFn.offset(t);
            int targetOffset = t * rows * scalarWidth * cols;
            for (int row = 0; row < rows; row++) {
                System.arraycopy(source,
                        sourceOffset + row * scalarWidth * leadingDimension,
                        copy,
                        targetOffset + row * scalarWidth * cols,
                        scalarWidth * cols);
            }
        }
        return copy;
    }

    private static boolean[] copyMissingFlags(StateSpaceModel model) {
        int kEndog = model.observationDimension();
        int nobs = model.observationCount();
        boolean[] copy = new boolean[kEndog * nobs];
        for (int t = 0; t < nobs; t++) {
            int offset = t * kEndog;
            for (int obs = 0; obs < kEndog; obs++) {
                copy[offset + obs] = model.isMissing(obs, t);
            }
        }
        return copy;
    }

    private static int[] copyMissingCounts(StateSpaceModel model) {
        int[] copy = new int[model.observationCount()];
        for (int t = 0; t < copy.length; t++) {
            copy[t] = model.missingCount(t);
        }
        return copy;
    }

    private static long doubleCount(double[] values) {
        return values == null ? 0L : values.length;
    }

    private static long intByteCount(int[] values) {
        return values == null ? 0L : (long) values.length * Integer.BYTES;
    }

    private static long booleanByteCount(boolean[] values) {
        return values == null ? 0L : values.length;
    }

    @FunctionalInterface
    private interface IntOffset {
        int offset(int t);
    }

    public static final class Builder {
        private final int kEndog;
        private final int kStates;
        private final int kPosdef;
        private final int nobs;
        private final int scalarWidth;

        private double[] design;
        private int designStride;
        private double[] obsIntercept;
        private int obsInterceptStride;
        private double[] obsCov;
        private int obsCovStride;
        private double[] transition;
        private int transitionStride;
        private double[] stateIntercept;
        private int stateInterceptStride;
        private double[] selection;
        private int selectionStride;
        private double[] stateCov;
        private int stateCovStride;
        private double[] endog;
        private int endogBase;
        private int endogStride;
        private boolean[] missing;
        private int[] nmissing;

        private Builder(int kEndog, int kStates, int kPosdef, int nobs, int scalarWidth) {
            validateDimensions(kEndog, kStates, kPosdef, nobs);
            this.kEndog = kEndog;
            this.kStates = kStates;
            this.kPosdef = kPosdef;
            this.nobs = nobs;
            this.scalarWidth = scalarWidth;
            this.endogStride = obsVectorBlockSize(kEndog, scalarWidth);
        }

        public Builder design(double[] values, boolean timeVarying) {
            this.design = values;
            this.designStride = timeVarying ? designBlockSize(kEndog, kStates, scalarWidth) : 0;
            return this;
        }

        public Builder obsIntercept(double[] values, boolean timeVarying) {
            this.obsIntercept = values;
            this.obsInterceptStride = timeVarying ? obsVectorBlockSize(kEndog, scalarWidth) : 0;
            return this;
        }

        public Builder obsCov(double[] values, boolean timeVarying) {
            this.obsCov = values;
            this.obsCovStride = timeVarying ? obsCovBlockSize(kEndog, scalarWidth) : 0;
            return this;
        }

        public Builder transition(double[] values, boolean timeVarying) {
            this.transition = values;
            this.transitionStride = timeVarying ? transitionBlockSize(kStates, scalarWidth) : 0;
            return this;
        }

        public Builder stateIntercept(double[] values, boolean timeVarying) {
            this.stateIntercept = values;
            this.stateInterceptStride = timeVarying ? stateVectorBlockSize(kStates, scalarWidth) : 0;
            return this;
        }

        public Builder selection(double[] values, boolean timeVarying) {
            this.selection = values;
            this.selectionStride = timeVarying ? selectionBlockSize(kStates, kPosdef, scalarWidth) : 0;
            return this;
        }

        public Builder stateCovariance(double[] values, boolean timeVarying) {
            this.stateCov = values;
            this.stateCovStride = timeVarying ? stateCovBlockSize(kPosdef, scalarWidth) : 0;
            return this;
        }

        public Builder stateCov(double[] values, boolean timeVarying) {
            return stateCovariance(values, timeVarying);
        }

        public Builder endog(double[] values) {
            return endog(values, 0);
        }

        public Builder endog(double[] values, int endogBase) {
            this.endog = values;
            this.endogBase = endogBase;
            this.endogStride = obsVectorBlockSize(kEndog, scalarWidth);
            return this;
        }

        public Builder missing(boolean[] missing) {
            this.missing = missing;
            this.nmissing = null;
            return this;
        }

        public Builder missing(boolean[] missing, int[] nmissing) {
            this.missing = missing;
            this.nmissing = nmissing;
            return this;
        }

        public Builder allObserved() {
            this.missing = null;
            this.nmissing = null;
            return this;
        }

        public StateSpaceModel build() {
            int obsVectorSize = obsVectorBlockSize(kEndog, scalarWidth);
            design = Objects.requireNonNull(design, "design");
            obsCov = Objects.requireNonNull(obsCov, "obsCov");
            transition = Objects.requireNonNull(transition, "transition");
            selection = Objects.requireNonNull(selection, "selection");
            stateCov = Objects.requireNonNull(stateCov, "stateCov");
            if (obsIntercept == null) {
                obsIntercept = new double[obsVectorSize];
                obsInterceptStride = 0;
            }
            if (stateIntercept == null) {
                stateIntercept = new double[stateVectorBlockSize(kStates, scalarWidth)];
                stateInterceptStride = 0;
            }
            return new StateSpaceModel(kEndog, kStates, kPosdef, nobs, scalarWidth,
                design, designStride,
                obsIntercept, obsInterceptStride,
                obsCov, obsCovStride,
                transition, transitionStride,
                stateIntercept, stateInterceptStride,
                selection, selectionStride,
                stateCov, stateCovStride,
                endog == null ? new double[obsVectorSize * nobs] : endog,
                endogBase,
                endogStride,
                missing,
                nmissing);
        }
    }
}