package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;

/**
 * Package-private unsafe view over a completed particle-filter run.
 *
 * <p>The arrays are owned by the {@link ParticleWorkspace} used for the run and
 * are valid only until that workspace is reused or closed.</p>
 */
final class BorrowedFilterResult<Y> {
    private final int N;
    private final int dim;
    private final int T;
    final double[] logLikelihoodSeries;
    final double[] essSeries;
    final boolean[] resampledFlags;
    final double[] finalParticles;
    final double[] finalLogWeights;
    final ParticleHistory particleHistory;
    final TransitionDensity<Y> transitionDensity;
    final RandomBatch rng;

    BorrowedFilterResult(int N,
                         int dim,
                         int T,
                         double[] logLikelihoodSeries,
                         double[] essSeries,
                         boolean[] resampledFlags,
                         double[] finalParticles,
                         double[] finalLogWeights,
                         ParticleHistory particleHistory,
                         TransitionDensity<Y> transitionDensity,
                         RandomBatch rng) {
        this.N = N;
        this.dim = dim;
        this.T = T;
        this.logLikelihoodSeries = logLikelihoodSeries;
        this.essSeries = essSeries;
        this.resampledFlags = resampledFlags;
        this.finalParticles = finalParticles;
        this.finalLogWeights = finalLogWeights;
        this.particleHistory = particleHistory;
        this.transitionDensity = transitionDensity;
        this.rng = rng;
    }

    int N() { return N; }

    int dim() { return dim; }

    int T() { return T; }

    double logLikelihoodAt(int t) { return logLikelihoodSeries[t]; }

    double essAt(int t) { return essSeries[t]; }

    boolean resampledAt(int t) { return resampledFlags[t]; }
}
