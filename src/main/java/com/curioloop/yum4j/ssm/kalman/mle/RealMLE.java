package com.curioloop.yum4j.ssm.kalman.mle;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;

import java.util.Arrays;

public abstract class RealMLE extends MLEModel<FilterResult, SmootherResult> {

    protected RealMLE(KalmanSSM templateModel,
                      double[] startParams,
                      String... parameterNames) {
        super(templateModel, startParams, parameterNames);
    }

    @Override
    protected final void requireSupported(KalmanSSM templateModel) {
        KalmanSSMSupport.requireRealStorage(templateModel, "templateModel");
    }

    @Override
    protected final EvaluationContext createEvaluationContext(KalmanSSM model) {
        return new RealEvaluationContext(model);
    }

    @Override
    protected final void resetEvaluationContext(EvaluationContext context) {
        RealEvaluationContext realContext = (RealEvaluationContext) context;
        realContext.workspace.releaseRetainedScratch();
        realContext.workspace.releaseRetainedResults();
    }

    @Override
    protected final FilterResult filter(EvaluationContext context,
                                        FilterOptions options) {
        return filterBorrowed(context, options).copy();
    }

    @Override
    protected final FilterResult filterBorrowed(EvaluationContext context,
                                                FilterOptions options) {
        RealEvaluationContext realContext = (RealEvaluationContext) context;
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        return KalmanEngine.filterBorrowedUnsafe(realContext.model, realContext.initialState,
            realContext.filterWorkspace,
            resolvedOptions);
    }

    @Override
    protected final SmootherResult smooth(EvaluationContext context,
                                          SmootherOptions options) {
        RealEvaluationContext realContext = (RealEvaluationContext) context;
        return SmootherEngine.smooth(realContext.model, realContext.initialState,
            realContext.workspace,
            options);
    }

    @Override
    protected final double logLikelihood(FilterResult filterResult) {
        return filterResult.logLikelihood();
    }

    @Override
    protected final double logLikelihood(FilterResult filterResult, int burn) {
        int start = Math.max(0, burn);
        if (start == 0) {
            return filterResult.logLikelihood();
        }
        double sum = 0.0;
        for (int t = Math.min(start, filterResult.nobs); t < filterResult.nobs; t++) {
            sum += filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    @Override
    protected final double[] logLikelihoodObs(FilterResult filterResult) {
        int base = filterResult.logLikelihoodObsBase();
        return Arrays.copyOfRange(filterResult.logLikelihoodObs, base, base + filterResult.nobs);
    }

    @Override
    protected final LogLikelihoodObsView logLikelihoodObsView(FilterResult filterResult) {
        return new LogLikelihoodObsView(filterResult.logLikelihoodObs, filterResult.logLikelihoodObsBase(), filterResult.nobs);
    }

    private static final class RealEvaluationContext extends EvaluationContext {
        private final KalmanEngine.Workspace filterWorkspace = KalmanEngine.workspace();
        private final SmootherEngine.Workspace workspace = SmootherEngine.workspace(filterWorkspace);

        private RealEvaluationContext(KalmanSSM model) {
            super(model);
        }
    }

}
