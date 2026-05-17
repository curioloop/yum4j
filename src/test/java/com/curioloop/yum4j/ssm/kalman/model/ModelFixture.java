package com.curioloop.yum4j.ssm.kalman.model;

import java.util.Objects;

public class ModelFixture extends KalmanSSM {

    private static final int REAL_SCALAR_WIDTH = 1;
    private static final int COMPLEX_SCALAR_WIDTH = 2;

    public final int kEndog;
    public final int kStates;
    public final int kPosdef;
    public final int nobs;

    private final int scalarWidth;

    protected int endogBase;

    public final boolean[] missing;
    public final int[] nmissing;

    public double[] design;
    public double[] obsIntercept;
    public double[] obsCov;
    public double[] transition;
    public double[] stateIntercept;
    public double[] selection;
    public double[] stateCov;
    public double[] endog;

    public ModelFixture(int kEndog, int kStates, int kPosdef, int nobs) {
        this(kEndog, kStates, kPosdef, nobs, REAL_SCALAR_WIDTH,
                new double[kEndog * REAL_SCALAR_WIDTH * kStates * nobs],
                new double[REAL_SCALAR_WIDTH * kEndog * nobs],
                new double[kEndog * REAL_SCALAR_WIDTH * kEndog * nobs],
                new double[kStates * REAL_SCALAR_WIDTH * kStates * nobs],
                new double[REAL_SCALAR_WIDTH * kStates * nobs],
                new double[kStates * REAL_SCALAR_WIDTH * kPosdef * nobs],
                new double[kPosdef * REAL_SCALAR_WIDTH * kPosdef * nobs],
                new double[REAL_SCALAR_WIDTH * kEndog * nobs],
                new boolean[kEndog * nobs],
                new int[nobs],
                0);
    }

    public static ModelFixture complex(int kEndog, int kStates, int kPosdef, int nobs) {
        return new ModelFixture(kEndog, kStates, kPosdef, nobs, COMPLEX_SCALAR_WIDTH,
                new double[kEndog * COMPLEX_SCALAR_WIDTH * kStates * nobs],
                new double[COMPLEX_SCALAR_WIDTH * kEndog * nobs],
                new double[kEndog * COMPLEX_SCALAR_WIDTH * kEndog * nobs],
                new double[kStates * COMPLEX_SCALAR_WIDTH * kStates * nobs],
                new double[COMPLEX_SCALAR_WIDTH * kStates * nobs],
                new double[kStates * COMPLEX_SCALAR_WIDTH * kPosdef * nobs],
                new double[kPosdef * COMPLEX_SCALAR_WIDTH * kPosdef * nobs],
                new double[COMPLEX_SCALAR_WIDTH * kEndog * nobs],
                new boolean[kEndog * nobs],
                new int[nobs],
                0);
    }

    private ModelFixture(int kEndog,
                         int kStates,
                         int kPosdef,
                         int nobs,
                         int scalarWidth,
                         double[] design,
                         double[] obsIntercept,
                         double[] obsCov,
                         double[] transition,
                         double[] stateIntercept,
                         double[] selection,
                         double[] stateCov,
                         double[] endog,
                         boolean[] missing,
                         int[] nmissing,
                         int endogBase) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.kPosdef = kPosdef;
        this.nobs = nobs;
        this.scalarWidth = scalarWidth;
        this.design = Objects.requireNonNull(design, "design");
        this.obsIntercept = Objects.requireNonNull(obsIntercept, "obsIntercept");
        this.obsCov = Objects.requireNonNull(obsCov, "obsCov");
        this.transition = Objects.requireNonNull(transition, "transition");
        this.stateIntercept = Objects.requireNonNull(stateIntercept, "stateIntercept");
        this.selection = Objects.requireNonNull(selection, "selection");
        this.stateCov = Objects.requireNonNull(stateCov, "stateCov");
        this.endog = Objects.requireNonNull(endog, "endog");
        this.missing = Objects.requireNonNull(missing, "missing");
        this.nmissing = Objects.requireNonNull(nmissing, "nmissing");
        this.endogBase = endogBase;
    }

    public static ModelFixture copyOf(KalmanSSM source) {
        Objects.requireNonNull(source, "source");
        if (source instanceof ModelFixture fixture) {
            return new ModelFixture(fixture.kEndog, fixture.kStates, fixture.kPosdef, fixture.nobs,
                fixture.scalarWidth,
                fixture.design.clone(),
                fixture.obsIntercept.clone(),
                fixture.obsCov.clone(),
                fixture.transition.clone(),
                fixture.stateIntercept.clone(),
                fixture.selection.clone(),
                fixture.stateCov.clone(),
                fixture.endog.clone(),
                fixture.missing.clone(),
                fixture.nmissing.clone(),
                0);
        }
        int scalarWidth = scalarWidth(source.complex());
        return new ModelFixture(source.observationDimension(),
            source.stateCount(),
            source.stateDisturbanceCount(),
            source.observationCount(),
                scalarWidth,
            copyDesign(source, scalarWidth),
            copyObservationVector(source.obsInterceptData(), source::obsInterceptOffset,
                source.observationDimension(), source.observationCount(), scalarWidth),
            copySquareBlocks(source.obsCovData(), source::obsCovOffset,
                source.obsCovLeadingDimension(), source.observationDimension(), source.observationCount(), scalarWidth),
            copySquareBlocks(source.transitionData(), source::transitionOffset,
                source.transitionLeadingDimension(), source.stateCount(), source.observationCount(), scalarWidth),
            copyStateVector(source.stateInterceptData(), source::stateInterceptOffset,
                source.stateCount(), source.observationCount(), scalarWidth),
            copyRectangularBlocks(source.selectionData(), source::selectionOffset,
                source.selectionLeadingDimension(), source.stateCount(), source.stateDisturbanceCount(), source.observationCount(), scalarWidth),
            copySquareBlocks(source.stateCovarianceData(), source::stateCovarianceOffset,
                source.stateCovarianceLeadingDimension(), source.stateDisturbanceCount(), source.observationCount(), scalarWidth),
            copyObservationVector(source.endogData(), source::endogOffset,
                source.observationDimension(), source.observationCount(), scalarWidth),
            copyMissingFlags(source),
            copyMissingCounts(source),
            0);
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

    @Override
    public int stateCount() {
        return kStates;
    }

    @Override
    public int stateDisturbanceCount() {
        return kPosdef;
    }

    @Override
    public int observationCount() {
        return nobs;
    }

    @Override
    public boolean complex() {
        return scalarWidth == COMPLEX_SCALAR_WIDTH;
    }

    @Override
    public double[] transitionData() {
        return transition;
    }

    @Override
    public int transitionOffset(int t) {
        return t * kStates * scalarWidth * kStates;
    }

    @Override
    public int transitionLeadingDimension() {
        return kStates;
    }

    @Override
    public double[] stateInterceptData() {
        return stateIntercept;
    }

    @Override
    public int stateInterceptOffset(int t) {
        return t * scalarWidth * kStates;
    }

    @Override
    public double[] selectionData() {
        return selection;
    }

    @Override
    public int selectionOffset(int t) {
        return t * kStates * scalarWidth * kPosdef;
    }

    @Override
    public int selectionLeadingDimension() {
        return kPosdef;
    }

    @Override
    public double[] stateCovarianceData() {
        return stateCov;
    }

    @Override
    public int stateCovarianceOffset(int t) {
        return t * kPosdef * scalarWidth * kPosdef;
    }

    public int stateCovOffset(int t) {
        return stateCovarianceOffset(t);
    }

    @Override
    public int stateCovarianceLeadingDimension() {
        return kPosdef;
    }

    @Override
    public int observationDimension() {
        return kEndog;
    }

    @Override
    public double[] designData() {
        return design;
    }

    @Override
    public int designOffset(int t) {
        return t * kEndog * scalarWidth * kStates;
    }

    @Override
    public int designLeadingDimension() {
        return kStates;
    }

    @Override
    public double[] obsInterceptData() {
        return obsIntercept;
    }

    @Override
    public int obsInterceptOffset(int t) {
        return t * scalarWidth * kEndog;
    }

    @Override
    public double[] obsCovData() {
        return obsCov;
    }

    @Override
    public int obsCovOffset(int t) {
        return t * kEndog * scalarWidth * kEndog;
    }

    @Override
    public int obsCovLeadingDimension() {
        return kEndog;
    }

    @Override
    public double[] endogData() {
        return endog;
    }

    @Override
    public int endogOffset(int t) {
        return endogBase + t * scalarWidth * kEndog;
    }

    @Override
    public boolean isMissing(int obsIndex, int t) {
        return missing[missingOffset(t) + obsIndex];
    }

    @Override
    public int missingCount(int t) {
        return nmissing[t];
    }

    private int missingOffset(int t) {
        return t * kEndog;
    }

    private long retainedMetadataByteCount() {
        return booleanByteCount(missing) + intByteCount(nmissing);
    }

    private static double[] copyDesign(KalmanSSM model, int scalarWidth) {
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

    private static boolean[] copyMissingFlags(KalmanSSM model) {
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

    private static int[] copyMissingCounts(KalmanSSM model) {
        int[] copy = new int[model.observationCount()];
        for (int t = 0; t < copy.length; t++) {
            copy[t] = model.missingCount(t);
        }
        return copy;
    }

    private static int scalarWidth(boolean complex) {
        return complex ? COMPLEX_SCALAR_WIDTH : REAL_SCALAR_WIDTH;
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
}