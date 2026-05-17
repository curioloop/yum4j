package com.curioloop.yum4j.ssm.kalman.mle;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.ZSmootherResult;

import java.util.Arrays;

public abstract class ComplexMLE extends MLEModel<ZFilterResult, ZSmootherResult> {

    protected ComplexMLE(KalmanSSM templateModel,
                         double[] startParams,
                         String... parameterNames) {
        super(templateModel, startParams, parameterNames);
    }

    @Override
    protected final void requireSupported(KalmanSSM templateModel) {
        KalmanSSMSupport.requireComplexStorage(templateModel, "templateModel");
    }

    @Override
    protected final EvaluationContext createEvaluationContext(KalmanSSM model) {
        return new ComplexEvaluationContext(model);
    }

    @Override
    protected final void resetEvaluationContext(EvaluationContext context) {
        ComplexEvaluationContext complexContext = (ComplexEvaluationContext) context;
        complexContext.workspace.releaseRetainedScratch();
        complexContext.workspace.releaseRetainedResults();
    }

    @Override
    protected final ZFilterResult filter(EvaluationContext context,
                                         FilterOptions options) {
        return filterBorrowed(context, options).copy();
    }

    @Override
    protected final ZFilterResult filterBorrowed(EvaluationContext context,
                                                 FilterOptions options) {
        ComplexEvaluationContext complexContext = (ComplexEvaluationContext) context;
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        return KalmanEngine.filterComplexBorrowedUnsafe(complexContext.model,
            complexContext.initialState,
            complexContext.filterWorkspace,
            resolvedOptions);
    }

    @Override
    protected final ZSmootherResult smooth(EvaluationContext context,
                                           SmootherOptions options) {
        ComplexEvaluationContext complexContext = (ComplexEvaluationContext) context;
        return SmootherEngine.smoothComplex(complexContext.model,
            complexContext.initialState,
            complexContext.workspace,
            options);
    }

    @Override
    protected final double logLikelihood(ZFilterResult filterResult) {
        return filterResult.logLikelihood();
    }

    @Override
    protected final double logLikelihood(ZFilterResult filterResult, int burn) {
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
    protected final double[] logLikelihoodObs(ZFilterResult filterResult) {
        int base = filterResult.logLikelihoodObsBase();
        return Arrays.copyOfRange(filterResult.logLikelihoodObs, base, base + filterResult.nobs);
    }

    @Override
    protected final LogLikelihoodObsView logLikelihoodObsView(ZFilterResult filterResult) {
        return new LogLikelihoodObsView(filterResult.logLikelihoodObs, filterResult.logLikelihoodObsBase(), filterResult.nobs);
    }

    private static final class ComplexEvaluationContext extends EvaluationContext {
        private final KalmanEngine.Workspace filterWorkspace = KalmanEngine.workspace();
        private final SmootherEngine.Workspace workspace = SmootherEngine.workspace(filterWorkspace);

        private ComplexEvaluationContext(KalmanSSM model) {
            super(model);
        }
    }

}
