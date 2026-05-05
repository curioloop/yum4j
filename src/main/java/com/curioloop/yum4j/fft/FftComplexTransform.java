package com.curioloop.yum4j.fft;

/**
 * Route selector for complex FFT transforms.
 *
 * <p>Chooses the optimal algorithm at construction time based on the transform
 * length: split-radix for powers of two, mixed-radix for smooth numbers, and
 * Bluestein convolution for large prime factors.</p>
 */
final class FftComplexTransform {

    /** Strategy that executes the actual FFT computation. */
    sealed interface Strategy {
        void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace);
        void exec(double[] data, int dataOffset, double factor, boolean forward);
        FftWorkspace.Requirement workspaceRequirement();
    }

    value record SplitRadix(FftSplitRadixPlan plan) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
            plan.exec(data, dataOffset, factor, forward);
        }
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward) {
            plan.exec(data, dataOffset, factor, forward);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return plan.workspaceRequirement();
        }
    }

    value record MixedRadix(FftComplexPlan plan) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
            plan.exec(data, dataOffset, factor, forward, workspace);
        }
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward) {
            FftWorkspace workspace = plan.workspaceRequirement().allocate();
            workspace.reset();
            plan.exec(data, dataOffset, factor, forward, workspace);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return plan.workspaceRequirement();
        }
    }

    value record Bluestein(FftBluesteinPlan plan) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
            plan.exec(data, dataOffset, factor, forward, workspace);
        }
        @Override public void exec(double[] data, int dataOffset, double factor, boolean forward) {
            FftWorkspace workspace = plan.workspaceRequirement().allocate();
            workspace.reset();
            plan.exec(data, dataOffset, factor, forward, workspace);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return plan.workspaceRequirement();
        }
    }

    private final int length;
    private final Strategy strategy;

    FftComplexTransform(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length FFT requested");
        }
        this.length = length;

        // Power-of-two: use split-radix (Ooura algorithm)
        if (length >= 2 && (length & (length - 1)) == 0) {
            this.strategy = new SplitRadix(FftPlanCache.splitRadixPlan(length));
            return;
        }

        long largestPrimeFactor = length < 50 ? 0 : FftUtil.largestPrimeFactor(length);
        if (largestPrimeFactor * largestPrimeFactor <= length) {
            this.strategy = new MixedRadix(new FftComplexPlan(length));
            return;
        }
        double directCost = FftUtil.costGuess(length);
        double blueCost = 1.5 * 2.0 * FftUtil.costGuess(FftUtil.goodSizeComplex(2L * length - 1L));
        if (blueCost < directCost) {
            this.strategy = new Bluestein(new FftBluesteinPlan(length));
        } else {
            this.strategy = new MixedRadix(new FftComplexPlan(length));
        }
    }

    void exec(double[] data, double factor, boolean forward) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, forward, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean forward) {
        if (dataOffset < 0 || data.length - dataOffset < 2 * length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        strategy.exec(data, dataOffset, factor, forward);
    }

    void exec(double[] data, double factor, boolean forward, FftWorkspace workspace) {
        exec(data, 0, factor, forward, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
        if (dataOffset < 0 || data.length - dataOffset < 2 * length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        strategy.exec(data, dataOffset, factor, forward, workspace);
    }

    FftWorkspace.Requirement workspaceRequirement() {
        return strategy.workspaceRequirement();
    }

    int length() {
        return length;
    }

    /** Package-private: used by test utilities to inspect the underlying plan. */
    Strategy strategy() {
        return strategy;
    }
}
