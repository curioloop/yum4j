package com.curioloop.yum4j.fft;

/**
 * Route selector for real FFT transforms.
 *
 * <p>Chooses the optimal algorithm at construction time based on the transform
 * length: split-radix for powers of two (≥4), mixed-radix for smooth numbers,
 * and Bluestein convolution for large prime factors.</p>
 */
final class FftRealTransform {

    /** Strategy that executes the actual real FFT computation. */
    sealed interface Strategy {
        void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace);
        FftWorkspace.Requirement workspaceRequirement();
    }

    value record SplitRadixReal(FftSplitRadixPlan plan, int length) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
            execSplitRadixReal(plan, length, data, dataOffset, factor, realToHalfComplex);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return FftWorkspace.Requirement.empty();
        }
    }

    value record MixedRadixReal(FftRealPlan plan) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
            plan.exec(data, dataOffset, factor, realToHalfComplex, workspace);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return plan.workspaceRequirement();
        }
    }

    value record BluesteinReal(FftBluesteinPlan plan) implements Strategy {
        @Override public void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
            plan.execReal(data, dataOffset, factor, realToHalfComplex, workspace);
        }
        @Override public FftWorkspace.Requirement workspaceRequirement() {
            return plan.workspaceRequirement();
        }
    }

    private final int length;
    private final Strategy strategy;

    FftRealTransform(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length FFT requested");
        }
        this.length = length;

        // Power-of-two (length >= 4): use split-radix
        if (length >= 4 && (length & (length - 1)) == 0) {
            this.strategy = new SplitRadixReal(FftPlanCache.splitRadixPlan(length), length);
            return;
        }

        long largestPrimeFactor = length < 50 ? 0 : FftUtil.largestPrimeFactor(length);
        if (largestPrimeFactor * largestPrimeFactor <= length) {
            this.strategy = new MixedRadixReal(new FftRealPlan(length));
            return;
        }
        double directCost = 0.5 * FftUtil.costGuess(length);
        double blueCost = 1.5 * 2.0 * FftUtil.costGuess(FftUtil.goodSizeComplex(2L * length - 1L));
        if (blueCost < directCost) {
            this.strategy = new BluesteinReal(new FftBluesteinPlan(length));
        } else {
            this.strategy = new MixedRadixReal(new FftRealPlan(length));
        }
    }

    void exec(double[] data, double factor, boolean realToHalfComplex) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, realToHalfComplex, workspace);
    }

    void exec(double[] data, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
        exec(data, 0, factor, realToHalfComplex, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
        if (dataOffset < 0 || data.length - dataOffset < length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        strategy.exec(data, dataOffset, factor, realToHalfComplex, workspace);
    }

    boolean tryRealToHermitianInPlace(double[] data, int dataOffset, double factor, boolean forward,
                                      FftWorkspace workspace) {
        if (strategy instanceof SplitRadixReal splitRadix) {
            int spectrumLength = 2 * (length / 2 + 1);
            if (dataOffset < 0 || data.length - dataOffset < spectrumLength) {
                throw new IllegalArgumentException("data length is smaller than Hermitian spectrum length");
            }
            execSplitRadixRealToHermitian(splitRadix.plan(), length, data, dataOffset, factor, forward);
            return true;
        }
        if (strategy instanceof BluesteinReal bluestein) {
            bluestein.plan().execRealToHermitian(data, dataOffset, factor, forward, workspace);
            return true;
        }
        return false;
    }

    FftWorkspace.Requirement workspaceRequirement() {
        return strategy.workspaceRequirement();
    }

    int length() {
        return length;
    }

    /** Package-private: used by test utilities to inspect the underlying strategy. */
    Strategy strategy() {
        return strategy;
    }

    // ---- split-radix real FFT implementation ----

    /**
     * Split-radix real FFT producing fftpack half-complex format.
     *
     * <p>JTransforms' split-radix realForward produces:
     * {@code a[0]=Re[0], a[1]=Re[n/2], a[2k]=Re[k], a[2k+1]=Im[k]} for k>=1.
     *
     * <p>Pocketfft's fftpack half-complex format is:
     * {@code a[0]=Re[0], a[1]=Re[1], a[2]=Im[1], a[3]=Re[2], a[4]=Im[2], ...}
     * with {@code a[n-1]=Im[n/2-1]} (n even) or the last real component packed at end.
     */
    private static void execSplitRadixReal(FftSplitRadixPlan splitRadixPlan, int n,
                                           double[] a, int offa, double factor, boolean realToHalfComplex) {
        int[] ip = splitRadixPlan.ip;
        double[] w = splitRadixPlan.w();
        int nw = splitRadixPlan.nw();
        int nc = splitRadixPlan.nc();

        if (realToHalfComplex) {
            if (n > 4) {
                FftSplitRadixPlan.cftfsub(n, a, offa, ip, nw, w);
                FftSplitRadixPlan.rftfsub(n, a, offa, nc, w, nw);
            } else {
                cftx020(a, offa);
            }
            double xi = a[offa] - a[offa + 1];
            a[offa] += a[offa + 1];
            a[offa + 1] = xi;
            a[offa] *= factor;
            double reNyquist = a[offa + 1] * factor;
            for (int i = 1; i < n - 1; i++) {
                a[offa + i] = a[offa + i + 1] * factor;
            }
            a[offa + n - 1] = reNyquist;
        } else {
            double fused = factor * 2.0;
            double reNyquist = a[offa + n - 1] * fused;
            for (int i = n - 1; i >= 2; i--) {
                a[offa + i] = a[offa + i - 1] * fused;
            }
            a[offa + 1] = reNyquist;
            a[offa] *= fused;
            a[offa + 1] = 0.5 * (a[offa] - a[offa + 1]);
            a[offa] -= a[offa + 1];
            if (n > 4) {
                FftSplitRadixPlan.rftfsub(n, a, offa, nc, w, nw);
                FftSplitRadixPlan.cftbsub(n, a, offa, ip, nw, w);
            } else {
                cftxc020(a, offa);
            }
        }
    }

    /**
     * Split-radix real FFT producing interleaved Hermitian complex output in place.
     *
     * <p>After Ooura's real forward kernel, non-DC bins are already laid out as
     * {@code Re[k], Im[k]} at {@code 2*k, 2*k+1}. The only special bins are DC and Nyquist,
     * so the standard r2c path can skip the FFTPACK half-complex compaction and later expand.</p>
     */
    private static void execSplitRadixRealToHermitian(FftSplitRadixPlan splitRadixPlan, int n,
                                                      double[] a, int offa, double factor, boolean forward) {
        int[] ip = splitRadixPlan.ip;
        double[] w = splitRadixPlan.w();
        int nw = splitRadixPlan.nw();
        int nc = splitRadixPlan.nc();

        if (n > 4) {
            FftSplitRadixPlan.cftfsub(n, a, offa, ip, nw, w);
            FftSplitRadixPlan.rftfsub(n, a, offa, nc, w, nw);
        } else {
            cftx020(a, offa);
        }

        double reZero = a[offa] + a[offa + 1];
        double reNyquist = a[offa] - a[offa + 1];
        double imaginaryFactor = forward ? factor : -factor;
        a[offa] = reZero * factor;
        a[offa + 1] = 0.0;
        for (int column = 1; column < n / 2; column++) {
            int index = offa + 2 * column;
            a[index] *= factor;
            a[index + 1] *= imaginaryFactor;
        }
        a[offa + n] = reNyquist * factor;
        a[offa + n + 1] = 0.0;
    }

    private static void cftx020(double[] a, int offa) {
        double x0r = a[offa] - a[offa + 2];
        double x0i = -a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] += a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    private static void cftxc020(double[] a, int offa) {
        double x0r = a[offa] - a[offa + 2];
        double x0i = a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] -= a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    // ---- static convenience methods ----

    static void realToHalfComplex(double[] data, double factor) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        FftPlanCache.realTransform(data.length).exec(data, factor, true);
    }

    static void halfComplexToReal(double[] data, double factor) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        FftPlanCache.realTransform(data.length).exec(data, factor, false);
    }

    static void roundTrip(double[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        FftRealTransform transform = FftPlanCache.realTransform(data.length);
        transform.exec(data, 1.0, true);
        transform.exec(data, 1.0 / data.length, false);
    }
}
