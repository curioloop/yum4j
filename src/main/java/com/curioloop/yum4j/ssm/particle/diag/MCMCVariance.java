package com.curioloop.yum4j.ssm.particle.diag;

/**
 * MCMC variance estimators: Geyer's initial-sequence estimator and
 * Tukey-Hanning spectral estimator.
 *
 * <p>These are utility methods (not a Collector) that operate on a
 * matrix of chain values shaped as {@code (P, M)} where P is the chain
 * length and M is the number of chains.
 *
 */
public final class MCMCVariance {

    private MCMCVariance() {}

    /**
     * Geyer's initial positive sequence estimator for the variance of
     * the chain mean.
     *
     * <p>Computes autocovariances and truncates the sum when a pair
     * {@code c[2k] + c[2k+1] < 0} (positive initial sequence rule).
     *
     * @param chain chain values of length P
     * @return estimated variance of the chain mean
     */
    public static double initialSequence(double[] chain) {
        int P = chain.length;
        if (P <= 1) return 0.0;
        int maxLag = P / 2;
        double[] gamma = new double[maxLag + 1];
        return initialSequence(chain, gamma);
    }

    /**
     * Internal: initial-sequence estimator reusing a pre-allocated gamma buffer.
     */
    static double initialSequence(double[] chain, double[] gamma) {
        int P = chain.length;
        if (P <= 1) return 0.0;

        double mean = 0.0;
        for (double v : chain) mean += v;
        mean /= P;

        // Compute autocovariances
        int maxLag = P / 2;
        for (int k = 0; k <= maxLag; k++) {
            double sum = 0.0;
            for (int t = 0; t < P - k; t++) {
                sum += (chain[t] - mean) * (chain[t + k] - mean);
            }
            gamma[k] = sum / P;
        }

        // Initial positive sequence: sum pairs until c[2k] + c[2k+1] < 0
        double varEst = gamma[0]; // c[0]
        for (int k = 1; k <= maxLag - 1; k += 2) {
            double pairSum = gamma[k] + gamma[k + 1];
            if (pairSum < 0) break;
            varEst += 2.0 * pairSum;
        }

        return varEst / P;
    }

    /**
     * Geyer's initial positive sequence estimator on a (P, M) matrix.
     * Averages the variance estimate across M chains.
     *
     * <p>Pre-allocates reusable buffers for both the chain extraction and
     * autocovariance computation to avoid per-chain allocation.
     *
     * @param values matrix of shape (P, M) stored row-major: values[p*M + m]
     * @param P      chain length
     * @param M      number of chains
     * @return estimated variance of the overall mean
     */
    public static double initialSequence(double[] values, int P, int M) {
        if (P <= 1 || M <= 0) return 0.0;

        double totalVar = 0.0;
        double[] chain = new double[P];
        int maxLag = P / 2;
        double[] gamma = new double[maxLag + 1];

        for (int m = 0; m < M; m++) {
            for (int p = 0; p < P; p++) {
                chain[p] = values[p * M + m];
            }
            totalVar += initialSequence(chain, gamma);
        }

        return totalVar / M;
    }

    /**
     * Tukey-Hanning spectral variance estimator.
     *
     * <p>Applies the spectral window
     * {@code w[k] = 1 − 2α + 2α·cos(πk/b)} with bandwidth
     * {@code b = c·√P} and α = 1/4.
     *
     * @param chain chain values of length P
     * @param c     bandwidth constant (typical: 1.0)
     * @return estimated variance of the chain mean
     */
    public static double tukeyHanning(double[] chain, double c) {
        int P = chain.length;
        if (P <= 1) return 0.0;
        int b = Math.max(1, (int) (c * Math.sqrt(P)));
        double[] gamma = new double[b + 1];
        return tukeyHanning(chain, c, gamma);
    }

    /**
     * Internal: Tukey-Hanning estimator reusing a pre-allocated gamma buffer.
     */
    static double tukeyHanning(double[] chain, double c, double[] gamma) {
        int P = chain.length;
        if (P <= 1) return 0.0;

        double mean = 0.0;
        for (double v : chain) mean += v;
        mean /= P;

        double alpha = 0.25;
        int b = Math.max(1, (int) (c * Math.sqrt(P)));

        // Compute autocovariances up to lag b
        for (int k = 0; k <= b; k++) {
            double sum = 0.0;
            for (int t = 0; t < P - k; t++) {
                sum += (chain[t] - mean) * (chain[t + k] - mean);
            }
            gamma[k] = sum / P;
        }

        // Apply Tukey-Hanning window
        double varEst = gamma[0];
        for (int k = 1; k <= b; k++) {
            double w = 1.0 - 2.0 * alpha + 2.0 * alpha * Math.cos(Math.PI * k / b);
            varEst += 2.0 * w * gamma[k];
        }

        return varEst / P;
    }

    /**
     * Tukey-Hanning estimator on a (P, M) matrix.
     *
     * <p>Pre-allocates reusable buffers for both the chain extraction and
     * autocovariance computation to avoid per-chain allocation.
     *
     * @param values matrix of shape (P, M) stored row-major
     * @param P      chain length
     * @param M      number of chains
     * @param c      bandwidth constant
     * @return estimated variance of the overall mean
     */
    public static double tukeyHanning(double[] values, int P, int M, double c) {
        if (P <= 1 || M <= 0) return 0.0;

        double totalVar = 0.0;
        double[] chain = new double[P];
        int b = Math.max(1, (int) (c * Math.sqrt(P)));
        double[] gamma = new double[b + 1];

        for (int m = 0; m < M; m++) {
            for (int p = 0; p < P; p++) {
                chain[p] = values[p * M + m];
            }
            totalVar += tukeyHanning(chain, c, gamma);
        }

        return totalVar / M;
    }
}
