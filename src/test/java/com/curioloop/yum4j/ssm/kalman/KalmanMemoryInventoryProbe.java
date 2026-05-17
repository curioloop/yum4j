package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.arena.FilterScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.SmootherScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.UnivariateScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.ZFilterScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.ZSmootherScratchLayout;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.tsa.sarimax.ARIMAOrder;
import com.curioloop.yum4j.tsa.sarimax.SARIMAX;
import com.curioloop.yum4j.tsa.sarimax.SARIMAXSpec;

import java.lang.management.ManagementFactory;
import java.util.Random;

public final class KalmanMemoryInventoryProbe {

    private KalmanMemoryInventoryProbe() {
    }

    public static void main(String[] args) {
        int nobs = args.length == 0 ? 32 : Integer.parseInt(args[0]);
        ModelFixture realModel = buildRealModel(nobs);
        ModelFixture collapsedModel = buildCollapsedEligibleModel(nobs);
        ModelFixture wideCollapsedModel = buildWideCollapsedEligibleModel(nobs);
        KalmanSSM chandrasekharModel = buildChandrasekharModel(nobs);
        ModelFixture complexModel = buildComplexModel(nobs);
        ModelFixture realTransformModel = buildRealObservationTransformModel(nobs);
        ModelFixture complexTransformModel = buildComplexObservationTransformModel(nobs);
        InitialState realInit = InitialState.approximateDiffuse();
        InitialState collapsedInit = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.0, 0.1, 0.1, 1.4});
        InitialState chandrasekharInit = InitialState.stationary(chandrasekharModel);
        InitialState complexInit = InitialState.approximateDiffuse();

        System.out.println("category,case,scratch,result,diffuse_surfaces_or_variates,nested_scratch,nested_result,total_bytes");
        recordModelStorageProfiles(realModel, complexModel);
        recordRealFilters(realModel, collapsedModel, chandrasekharModel, realInit, collapsedInit, chandrasekharInit);
        recordComplexFilters(complexModel, complexInit);
        recordFilterWorkspaceReuse(realModel, collapsedModel, chandrasekharModel, complexModel,
            realInit, collapsedInit, chandrasekharInit, complexInit);
        recordCopiedBorrowedApiProfiles(realModel, complexModel, realInit, complexInit);
        recordCollapsedOptionProfiles(collapsedModel, wideCollapsedModel, collapsedInit);
        recordObservationTransforms(realTransformModel, complexTransformModel, realInit, complexInit);
        recordSmoothers(realModel, complexModel, realInit, complexInit);
        recordSimulationSmoothers(realModel, complexModel, realInit, complexInit);
        System.out.println();
        System.out.println("allocation,case,iterations,total_allocated_bytes,bytes_per_iteration");
        recordAllocationProfiles(nobs);
        System.out.println();
        System.out.println("layout,case,slice,base,length,active,total_length,expected_optimized_length");
        recordScratchLayouts(realModel, complexModel);
    }

    private static void recordModelStorageProfiles(ModelFixture realModel, ModelFixture complexModel) {
        printModelProfile("real-template-model", realModel);
        printModelProfile("real-copy-model", KalmanSSM.copyOf(realModel));
        printModelProfile("complex-template-model", complexModel);
        printModelProfile("complex-copy-model", KalmanSSM.copyOf(complexModel));
    }

    private static void printModelProfile(String name, KalmanSSM model) {
        long retainedDoubles = model.retainedModelDoubleCount();
        long metadataBytes = model.retainedTotalByteCount() - retainedDoubles * Double.BYTES;
        print("model", name, retainedDoubles, 0L, metadataBytes, 0L, 0L, model.retainedTotalByteCount());
    }

    private static void recordAllocationProfiles(int nobs) {
        ModelFixture realModel = buildRealModel(nobs);
        ModelFixture complexModel = buildComplexModel(nobs);
        SARIMAX sarimax = buildProbeSarimax(nobs);
        double[] sarimaxParams = sarimax.startParams();
        FilterOptions likelihood = FilterOptions.likelihoodOnly();

        printAllocation("state-space-copy-real", 50,
            measureAllocatedBytes(50, () -> KalmanSSM.copyOf(realModel)));
        printAllocation("state-space-copy-complex", 50,
            measureAllocatedBytes(50, () -> KalmanSSM.copyOf(complexModel)));
        printAllocation("sarimax-loglik", 50,
            measureAllocatedBytes(50, () -> sarimax.logLikelihood(sarimaxParams, likelihood)));
        printAllocation("sarimax-loglik-obs", 25,
            measureAllocatedBytes(25, () -> sarimax.logLikelihoodObs(sarimaxParams, likelihood)));
        printAllocation("sarimax-snapshot-model", 25,
            measureAllocatedBytes(25, () -> sarimax.snapshotModel(sarimaxParams)));
        printAllocation("sarimax-score-obs", 5,
            measureAllocatedBytes(5, () -> sarimax.scoreObs(sarimaxParams)));
    }

    private static SARIMAX buildProbeSarimax(int nobs) {
        double[] endog = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            endog[t] = 0.4 + 0.03 * t + 0.12 * Math.sin(0.2 * t);
        }
        return new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 1), endog)
            .measurementError(true)
            .build());
    }

    private static long measureAllocatedBytes(int iterations, Runnable action) {
        if (iterations <= 0) {
            return 0L;
        }
        action.run();
        java.lang.management.ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        if (!(bean instanceof com.sun.management.ThreadMXBean allocationBean)
                || !allocationBean.isThreadAllocatedMemorySupported()) {
            long usedBefore = usedMemory();
            for (int i = 0; i < iterations; i++) {
                action.run();
            }
            long usedAfter = usedMemory();
            return Math.max(0L, usedAfter - usedBefore);
        }
        if (!allocationBean.isThreadAllocatedMemoryEnabled()) {
            allocationBean.setThreadAllocatedMemoryEnabled(true);
        }
        long threadId = Thread.currentThread().threadId();
        long before = allocationBean.getThreadAllocatedBytes(threadId);
        for (int i = 0; i < iterations; i++) {
            action.run();
        }
        long after = allocationBean.getThreadAllocatedBytes(threadId);
        return before < 0L || after < before ? -1L : after - before;
    }

    private static long usedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static void printAllocation(String name, int iterations, long allocatedBytes) {
        long perIteration = allocatedBytes < 0L ? -1L : allocatedBytes / Math.max(1, iterations);
        System.out.printf("allocation,%s,%d,%d,%d%n", name, iterations, allocatedBytes, perIteration);
    }

    private static void recordRealFilters(ModelFixture realModel,
                                          ModelFixture collapsedModel,
                                          KalmanSSM chandrasekharModel,
                                          InitialState realInit,
                                          InitialState collapsedInit,
                                          InitialState chandrasekharInit) {
        FilterOptions conventional = FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build();
        FilterOptions univariate = FilterOptions.builder().method(FilterMethod.UNIVARIATE).build();
        FilterOptions collapsed = FilterOptions.builder().method(FilterMethod.COLLAPSED).build();
        FilterOptions chandrasekhar = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build();

        KalmanEngine.Workspace conventionalWorkspace = KalmanEngine.workspace();
        FilterResult conventionalResult = KalmanEngine.filterBorrowedUnsafe(realModel, realInit, conventionalWorkspace, conventional);
        print("filter", "real-conventional", conventionalWorkspace.retainedFilterScratchDoubleCount(),
            conventionalWorkspace.retainedFilterResultDoubleCount(), diffuseSurfaceDoubles(conventionalResult), 0L, 0L,
            conventionalWorkspace.retainedFilterTotalByteCount());

        KalmanEngine.Workspace conventionalDiffuseWorkspace = KalmanEngine.workspace();
        FilterResult conventionalDiffuseResult = KalmanEngine.filterBorrowedUnsafe(realModel,
            InitialState.diffuse(), conventionalDiffuseWorkspace, conventional);
        print("filter", "real-conventional-exact-diffuse", conventionalDiffuseWorkspace.retainedFilterScratchDoubleCount(),
            conventionalDiffuseWorkspace.retainedFilterResultDoubleCount(), diffuseSurfaceDoubles(conventionalDiffuseResult), 0L, 0L,
            conventionalDiffuseWorkspace.retainedFilterTotalByteCount());

        KalmanEngine.Workspace univariateWorkspace = KalmanEngine.workspace();
        FilterResult univariateResult = KalmanEngine.filterBorrowedUnsafe(realModel, realInit, univariateWorkspace, univariate);
        print("filter", "real-univariate", univariateWorkspace.retainedUnivariateScratchDoubleCount(),
            univariateWorkspace.retainedUnivariateResultDoubleCount(), diffuseSurfaceDoubles(univariateResult), 0L, 0L,
            univariateWorkspace.retainedUnivariateTotalByteCount());

        KalmanEngine.Workspace univariateDiffuseWorkspace = KalmanEngine.workspace();
        FilterResult univariateDiffuseResult = KalmanEngine.filterBorrowedUnsafe(realModel,
            InitialState.diffuse(), univariateDiffuseWorkspace, univariate);
        print("filter", "real-univariate-exact-diffuse", univariateDiffuseWorkspace.retainedUnivariateScratchDoubleCount(),
            univariateDiffuseWorkspace.retainedUnivariateResultDoubleCount(), diffuseSurfaceDoubles(univariateDiffuseResult), 0L, 0L,
            univariateDiffuseWorkspace.retainedUnivariateTotalByteCount());

        KalmanEngine.Workspace collapsedWorkspace = KalmanEngine.workspace();
        FilterResult collapsedResult = KalmanEngine.filterBorrowedUnsafe(collapsedModel, collapsedInit, collapsedWorkspace, collapsed);
        print("filter", "real-collapsed", collapsedWorkspace.retainedCollapsedScratchDoubleCount(),
            collapsedWorkspace.retainedCollapsedResultDoubleCount(), diffuseSurfaceDoubles(collapsedResult), 0L, 0L,
            collapsedWorkspace.retainedCollapsedTotalByteCount());

        KalmanEngine.Workspace chandrasekharWorkspace = KalmanEngine.workspace();
        FilterResult chandrasekharResult = KalmanEngine.filterBorrowedUnsafe(chandrasekharModel, chandrasekharInit, chandrasekharWorkspace, chandrasekhar);
        print("filter", "real-chandrasekhar", chandrasekharWorkspace.retainedChandrasekharScratchDoubleCount(),
            chandrasekharWorkspace.retainedChandrasekharResultDoubleCount(), diffuseSurfaceDoubles(chandrasekharResult), 0L, 0L,
            chandrasekharWorkspace.retainedChandrasekharTotalByteCount());
    }

    private static void recordComplexFilters(ModelFixture complexModel, InitialState complexInit) {
        FilterOptions conventional = FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build();
        FilterOptions univariate = FilterOptions.builder().method(FilterMethod.UNIVARIATE).build();

        KalmanEngine.Workspace conventionalWorkspace = KalmanEngine.workspace();
        ZFilterResult conventionalResult = KalmanEngine.filterComplexBorrowedUnsafe(complexModel, complexInit, conventionalWorkspace, conventional);
        print("filter", "complex-conventional", conventionalWorkspace.retainedComplexFilterScratchDoubleCount(),
            conventionalWorkspace.retainedComplexFilterResultDoubleCount(), diffuseSurfaceDoubles(conventionalResult), 0L, 0L,
            conventionalWorkspace.retainedComplexFilterTotalByteCount());

        KalmanEngine.Workspace conventionalDiffuseWorkspace = KalmanEngine.workspace();
        ZFilterResult conventionalDiffuseResult = KalmanEngine.filterComplexBorrowedUnsafe(complexModel,
            InitialState.diffuse(), conventionalDiffuseWorkspace, conventional);
        print("filter", "complex-conventional-exact-diffuse", conventionalDiffuseWorkspace.retainedComplexFilterScratchDoubleCount(),
            conventionalDiffuseWorkspace.retainedComplexFilterResultDoubleCount(), diffuseSurfaceDoubles(conventionalDiffuseResult), 0L, 0L,
            conventionalDiffuseWorkspace.retainedComplexFilterTotalByteCount());

        KalmanEngine.Workspace univariateWorkspace = KalmanEngine.workspace();
        ZFilterResult univariateResult = KalmanEngine.filterComplexBorrowedUnsafe(complexModel, complexInit, univariateWorkspace, univariate);
        print("filter", "complex-univariate", univariateWorkspace.retainedComplexFilterScratchDoubleCount(),
            univariateWorkspace.retainedComplexFilterResultDoubleCount(), diffuseSurfaceDoubles(univariateResult), 0L, 0L,
            univariateWorkspace.retainedComplexFilterTotalByteCount());

        KalmanEngine.Workspace univariateDiffuseWorkspace = KalmanEngine.workspace();
        ZFilterResult univariateDiffuseResult = KalmanEngine.filterComplexBorrowedUnsafe(complexModel,
            InitialState.diffuse(), univariateDiffuseWorkspace, univariate);
        print("filter", "complex-univariate-exact-diffuse", univariateDiffuseWorkspace.retainedComplexFilterScratchDoubleCount(),
            univariateDiffuseWorkspace.retainedComplexFilterResultDoubleCount(), diffuseSurfaceDoubles(univariateDiffuseResult), 0L, 0L,
            univariateDiffuseWorkspace.retainedComplexFilterTotalByteCount());
    }

    private static void recordFilterWorkspaceReuse(ModelFixture realModel,
                                                   ModelFixture collapsedModel,
                                                   KalmanSSM chandrasekharModel,
                                                   ModelFixture complexModel,
                                                   InitialState realInit,
                                                   InitialState collapsedInit,
                                                   InitialState chandrasekharInit,
                                                   InitialState complexInit) {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();
        KalmanEngine.filterBorrowedUnsafe(realModel, realInit, workspace,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        KalmanEngine.filterBorrowedUnsafe(realModel, realInit, workspace,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());
        KalmanEngine.filterBorrowedUnsafe(collapsedModel, collapsedInit, workspace,
            FilterOptions.builder().method(FilterMethod.COLLAPSED).build());
        KalmanEngine.filterBorrowedUnsafe(chandrasekharModel, chandrasekharInit, workspace,
            FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build());
        print("workspace", "filter-real-multi-route", workspace.retainedRealFilterScratchDoubleCount(),
            workspace.retainedRealFilterResultDoubleCount(), 0L, 0L, 0L,
            workspace.retainedRealFilterTotalByteCount());

        KalmanEngine.filterComplexBorrowedUnsafe(complexModel, complexInit, workspace,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        KalmanEngine.filterComplexBorrowedUnsafe(complexModel, complexInit, workspace,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());
        print("workspace", "filter-real-plus-complex", workspace.retainedFilterWorkspaceScratchDoubleCount(),
            workspace.retainedFilterWorkspaceResultDoubleCount(), 0L, 0L, 0L,
            workspace.retainedFilterWorkspaceTotalByteCount());
    }

    private static void recordCopiedBorrowedApiProfiles(ModelFixture realModel,
                                                        ModelFixture complexModel,
                                                        InitialState realInit,
                                                        InitialState complexInit) {
        FilterOptions filterOptions = FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build();

        KalmanEngine.Workspace realBorrowedWorkspace = KalmanEngine.workspace();
        KalmanEngine.filterBorrowedUnsafe(realModel, realInit, realBorrowedWorkspace, filterOptions);
        print("api", "filter-real-borrowed-api", realBorrowedWorkspace.retainedFilterScratchDoubleCount(),
            realBorrowedWorkspace.retainedFilterResultDoubleCount(), 0L, 0L, 0L,
            realBorrowedWorkspace.retainedFilterTotalByteCount());

        KalmanEngine.Workspace realCopiedWorkspace = KalmanEngine.workspace();
        KalmanEngine.builder(realModel)
            .initialState(realInit)
            .workspace(realCopiedWorkspace)
            .options(filterOptions)
            .filter();
        print("api", "filter-real-copied-api", realCopiedWorkspace.retainedFilterScratchDoubleCount(),
            realCopiedWorkspace.retainedFilterResultDoubleCount(), 0L, 0L, 0L,
            realCopiedWorkspace.retainedFilterTotalByteCount());

        KalmanEngine.Workspace complexBorrowedWorkspace = KalmanEngine.workspace();
        KalmanEngine.filterComplexBorrowedUnsafe(complexModel, complexInit, complexBorrowedWorkspace, filterOptions);
        print("api", "filter-complex-borrowed-api", complexBorrowedWorkspace.retainedComplexFilterScratchDoubleCount(),
            complexBorrowedWorkspace.retainedComplexFilterResultDoubleCount(), 0L, 0L, 0L,
            complexBorrowedWorkspace.retainedComplexFilterTotalByteCount());

        KalmanEngine.Workspace complexCopiedWorkspace = KalmanEngine.workspace();
        KalmanEngine.builder(complexModel)
            .initialState(complexInit)
            .workspace(complexCopiedWorkspace)
            .options(filterOptions)
            .filterComplex();
        print("api", "filter-complex-copied-api", complexCopiedWorkspace.retainedComplexFilterScratchDoubleCount(),
            complexCopiedWorkspace.retainedComplexFilterResultDoubleCount(), 0L, 0L, 0L,
            complexCopiedWorkspace.retainedComplexFilterTotalByteCount());

        SmootherOptions smootherOptions = SmootherOptions.conventional();
        SmootherEngine.Workspace realSmootherBorrowedWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothBorrowedUnsafe(realModel, realInit, realSmootherBorrowedWorkspace, smootherOptions);
        print("api", "smoother-real-borrowed-api", realSmootherBorrowedWorkspace.retainedSmootherScratchDoubleCount(),
            realSmootherBorrowedWorkspace.retainedSmootherResultDoubleCount(), 0L,
            realSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            realSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            realSmootherBorrowedWorkspace.retainedSmootherTotalByteCount()
                + realSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceTotalByteCount());

        SmootherEngine.Workspace realSmootherCopiedWorkspace = SmootherEngine.workspace();
        SmootherEngine.smooth(realModel, realInit, realSmootherCopiedWorkspace, smootherOptions);
        print("api", "smoother-real-copied-api", realSmootherCopiedWorkspace.retainedSmootherScratchDoubleCount(),
            realSmootherCopiedWorkspace.retainedSmootherResultDoubleCount(), 0L,
            realSmootherCopiedWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            realSmootherCopiedWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            realSmootherCopiedWorkspace.retainedSmootherTotalByteCount()
                + realSmootherCopiedWorkspace.retainedNestedFilterWorkspaceTotalByteCount());

        SmootherEngine.Workspace complexSmootherBorrowedWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothComplexBorrowedUnsafe(complexModel, complexInit, complexSmootherBorrowedWorkspace, smootherOptions);
        print("api", "smoother-complex-borrowed-api", complexSmootherBorrowedWorkspace.retainedComplexSmootherScratchDoubleCount(),
            complexSmootherBorrowedWorkspace.retainedComplexSmootherResultDoubleCount(), 0L,
            complexSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            complexSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            complexSmootherBorrowedWorkspace.retainedComplexSmootherTotalByteCount()
                + complexSmootherBorrowedWorkspace.retainedNestedFilterWorkspaceTotalByteCount());

        SmootherEngine.Workspace complexSmootherCopiedWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothComplex(complexModel, complexInit, complexSmootherCopiedWorkspace, smootherOptions);
        print("api", "smoother-complex-copied-api", complexSmootherCopiedWorkspace.retainedComplexSmootherScratchDoubleCount(),
            complexSmootherCopiedWorkspace.retainedComplexSmootherResultDoubleCount(), 0L,
            complexSmootherCopiedWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            complexSmootherCopiedWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            complexSmootherCopiedWorkspace.retainedComplexSmootherTotalByteCount()
                + complexSmootherCopiedWorkspace.retainedNestedFilterWorkspaceTotalByteCount());
    }

    private static void recordCollapsedOptionProfiles(ModelFixture collapsedModel,
                                                      ModelFixture wideCollapsedModel,
                                                      InitialState collapsedInit) {
        recordCollapsedProfile("real-collapsed-likelihood-only", collapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainOnly(FilterOptions.Surface.LIKELIHOOD)
                .build());
        recordCollapsedProfile("real-collapsed-forecast-only", collapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainOnly(FilterOptions.Surface.FORECAST_MEAN,
                    FilterOptions.Surface.FORECAST_ERROR,
                    FilterOptions.Surface.FORECAST_COVARIANCE,
                    FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
                .build());
        recordCollapsedProfile("real-collapsed-full", collapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainAll()
                .build());
        recordCollapsedProfile("real-collapsed-wide-likelihood-only", wideCollapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainOnly(FilterOptions.Surface.LIKELIHOOD)
                .build());
        recordCollapsedProfile("real-collapsed-wide-forecast-only", wideCollapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainOnly(FilterOptions.Surface.FORECAST_MEAN,
                    FilterOptions.Surface.FORECAST_ERROR,
                    FilterOptions.Surface.FORECAST_COVARIANCE,
                    FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
                .build());
        recordCollapsedProfile("real-collapsed-wide-full", wideCollapsedModel, collapsedInit,
            FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainAll()
                .build());
    }

    private static void recordCollapsedProfile(String name,
                                               ModelFixture model,
                                               InitialState init,
                                               FilterOptions options) {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();
        FilterResult result = KalmanEngine.filterBorrowedUnsafe(model, init, workspace, options);
        print("filter", name, workspace.retainedCollapsedScratchDoubleCount(),
            workspace.retainedCollapsedResultDoubleCount(), diffuseSurfaceDoubles(result), 0L, 0L,
            workspace.retainedCollapsedTotalByteCount());
    }

    private static void recordObservationTransforms(ModelFixture realTransformModel,
                                                    ModelFixture complexTransformModel,
                                                    InitialState realInit,
                                                    InitialState complexInit) {
        FilterOptions univariate = FilterOptions.builder().method(FilterMethod.UNIVARIATE).build();
        KalmanEngine.Workspace realWorkspace = KalmanEngine.workspace();
        FilterResult realResult = KalmanEngine.filterBorrowedUnsafe(realTransformModel, realInit, realWorkspace, univariate);
        print("transform", "real-univariate-nondiagonal-observation", realWorkspace.retainedUnivariateScratchDoubleCount(),
            realWorkspace.retainedUnivariateResultDoubleCount(), diffuseSurfaceDoubles(realResult), 0L, 0L,
            realWorkspace.retainedUnivariateTotalByteCount());

        KalmanEngine.Workspace complexWorkspace = KalmanEngine.workspace();
        ZFilterResult complexResult = KalmanEngine.filterComplexBorrowedUnsafe(complexTransformModel, complexInit, complexWorkspace, univariate);
        print("transform", "complex-univariate-nondiagonal-observation", complexWorkspace.retainedComplexFilterScratchDoubleCount(),
            complexWorkspace.retainedComplexFilterResultDoubleCount(), diffuseSurfaceDoubles(complexResult), 0L, 0L,
            complexWorkspace.retainedComplexFilterTotalByteCount());
    }

    private static void recordSmoothers(ModelFixture realModel,
                                        ModelFixture complexModel,
                                        InitialState realInit,
                                        InitialState complexInit) {
        SmootherOptions smootherOptions = SmootherOptions.conventional();
        FilterResult realFilter = KalmanEngine.filter(realModel, realInit, smootherOptions.requiredFilterOptions());
        SmootherEngine.Workspace realWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothBorrowedUnsafe(realModel, realFilter, realWorkspace, smootherOptions);
        print("smoother", "real-conventional", realWorkspace.retainedSmootherScratchDoubleCount(),
            realWorkspace.retainedSmootherResultDoubleCount(), 0L, 0L, 0L,
            realWorkspace.retainedSmootherTotalByteCount());
        SmootherEngine.Workspace realPipelineWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothBorrowedUnsafe(realModel, realInit, realPipelineWorkspace, smootherOptions);
        print("smoother", "real-conventional-pipeline", realPipelineWorkspace.retainedSmootherScratchDoubleCount(),
            realPipelineWorkspace.retainedSmootherResultDoubleCount(), 0L,
            realPipelineWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            realPipelineWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            realPipelineWorkspace.retainedSmootherTotalByteCount()
                + realPipelineWorkspace.retainedNestedFilterWorkspaceTotalByteCount());

        ZFilterResult complexFilter = KalmanEngine.filterComplex(complexModel, complexInit, smootherOptions.requiredFilterOptions());
        SmootherEngine.Workspace complexWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothComplexBorrowedUnsafe(complexModel, complexFilter, complexWorkspace, smootherOptions);
        print("smoother", "complex-conventional", complexWorkspace.retainedComplexSmootherScratchDoubleCount(),
            complexWorkspace.retainedComplexSmootherResultDoubleCount(), 0L, 0L, 0L,
            complexWorkspace.retainedComplexSmootherTotalByteCount());
        SmootherEngine.Workspace complexPipelineWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothComplexBorrowedUnsafe(complexModel, complexInit, complexPipelineWorkspace, smootherOptions);
        print("smoother", "complex-conventional-pipeline", complexPipelineWorkspace.retainedComplexSmootherScratchDoubleCount(),
            complexPipelineWorkspace.retainedComplexSmootherResultDoubleCount(), 0L,
            complexPipelineWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            complexPipelineWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            complexPipelineWorkspace.retainedComplexSmootherTotalByteCount()
                + complexPipelineWorkspace.retainedNestedFilterWorkspaceTotalByteCount());

        SmootherEngine.Workspace sharedWorkspace = SmootherEngine.workspace();
        SmootherEngine.smoothBorrowedUnsafe(realModel, realInit, sharedWorkspace, smootherOptions);
        SmootherEngine.smoothComplexBorrowedUnsafe(complexModel, complexInit, sharedWorkspace, smootherOptions);
        print("workspace", "smoother-real-plus-complex", sharedWorkspace.retainedSmootherWorkspaceScratchDoubleCount(),
            sharedWorkspace.retainedSmootherWorkspaceResultDoubleCount(),
            0L,
            sharedWorkspace.retainedNestedFilterWorkspaceScratchDoubleCount(),
            sharedWorkspace.retainedNestedFilterWorkspaceResultDoubleCount(),
            sharedWorkspace.retainedSmootherWorkspaceTotalByteCount()
                + sharedWorkspace.retainedNestedFilterWorkspaceTotalByteCount());
    }

    private static void recordSimulationSmoothers(ModelFixture realModel,
                                                  ModelFixture complexModel,
                                                  InitialState realInit,
                                                  InitialState complexInit) {
        SimulationSmootherOptions options = SimulationSmootherOptions.defaults();
        recordRealSimulationProfile("real-kfs", realModel, realInit, options);
        recordRealSimulationCopiedProfile("real-kfs-copied-api", realModel, realInit, options);
        recordRealSimulationRepeatedProfile("real-kfs-repeated-borrowed-api", realModel, realInit, options);
        recordRealSimulationProfile("real-kfs-state-only", realModel, realInit, SimulationSmootherOptions.stateOnly());
        recordRealSimulationProfile("real-kfs-disturbance-only", realModel, realInit, SimulationSmootherOptions.disturbanceOnly());
        recordRealSimulationProfile("real-kfs-generated", realModel, realInit, options.withGeneratedOutputs());
        recordRealSimulationProfile("real-cfa-state-only", realModel, knownRealInitial(realModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordRealSimulationCopiedProfile("real-cfa-copied-api", realModel, knownRealInitial(realModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordRealSimulationRepeatedProfile("real-cfa-repeated-borrowed-api", realModel, knownRealInitial(realModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordRealSimulationRetainedCfaProfile("real-cfa-retained-posterior-api", realModel, knownRealInitial(realModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));

        recordComplexSimulationProfile("complex-kfs", complexModel, complexInit, options);
        recordComplexSimulationCopiedProfile("complex-kfs-copied-api", complexModel, complexInit, options);
        recordComplexSimulationRepeatedProfile("complex-kfs-repeated-borrowed-api", complexModel, complexInit, options);
        recordComplexSimulationProfile("complex-kfs-state-only", complexModel, complexInit, SimulationSmootherOptions.stateOnly());
        recordComplexSimulationProfile("complex-kfs-generated", complexModel, complexInit, options.withGeneratedOutputs());
        recordComplexSimulationProfile("complex-cfa-state-only", complexModel, knownComplexInitial(complexModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordComplexSimulationCopiedProfile("complex-cfa-copied-api", complexModel, knownComplexInitial(complexModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordComplexSimulationRepeatedProfile("complex-cfa-repeated-borrowed-api", complexModel, knownComplexInitial(complexModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
        recordComplexSimulationRetainedCfaProfile("complex-cfa-retained-posterior-api", complexModel, knownComplexInitial(complexModel),
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));
    }

    private static void recordRealSimulationProfile(String name,
                                                    ModelFixture realModel,
                                                    InitialState realInit,
                                                    SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        SimulationSmootherEngine.simulateBorrowedUnsafe(realModel, realInit,
            realVariates(realModel.nobs * realModel.kEndog, 7L),
            realVariates(realModel.nobs * realModel.kPosdef, 11L),
            realVariates(realModel.kStates, 13L),
            workspace,
            options);
        print("simulation", name, workspace.retainedWorkspaceDoubleCount(),
            workspace.retainedResultDoubleCount(), workspace.retainedVariateDoubleCount(),
            workspace.retainedNestedScratchDoubleCount(), workspace.retainedNestedResultDoubleCount(),
            workspace.retainedTotalByteCount());
        printRealSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordRealSimulationRepeatedProfile(String name,
                                                            ModelFixture realModel,
                                                            InitialState realInit,
                                                            SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        workspace.retainReusableWorkspaces();
        for (int i = 0; i < 2; i++) {
            SimulationSmootherEngine.simulateBorrowedUnsafe(realModel, realInit,
                realVariates(realModel.nobs * realModel.kEndog, 7L + i),
                realVariates(realModel.nobs * realModel.kPosdef, 11L + i),
                realVariates(realModel.kStates, 13L + i),
                workspace,
                options);
        }
        print("simulation", name, workspace.retainedWorkspaceDoubleCount(),
            workspace.retainedResultDoubleCount(), workspace.retainedVariateDoubleCount(),
            workspace.retainedNestedScratchDoubleCount(), workspace.retainedNestedResultDoubleCount(),
            workspace.retainedTotalByteCount());
        printRealSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordRealSimulationRetainedCfaProfile(String name,
                                                               ModelFixture realModel,
                                                               InitialState realInit,
                                                               SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        workspace.retainCfaPosteriorWorkspace();
        for (int i = 0; i < 2; i++) {
            SimulationSmootherEngine.simulateBorrowedUnsafe(realModel, realInit,
                realVariates(realModel.nobs * realModel.kEndog, 7L + i),
                realVariates(realModel.nobs * realModel.kPosdef, 11L + i),
                realVariates(realModel.kStates, 13L + i),
                workspace,
                options);
        }
        print("simulation", name, workspace.retainedWorkspaceDoubleCount(),
            workspace.retainedResultDoubleCount(), workspace.retainedVariateDoubleCount(),
            workspace.retainedNestedScratchDoubleCount(), workspace.retainedNestedResultDoubleCount(),
            workspace.retainedTotalByteCount());
        printRealSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordRealSimulationCopiedProfile(String name,
                                                          ModelFixture realModel,
                                                          InitialState realInit,
                                                          SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        SimulationSmootherEngine.simulate(realModel, realInit,
            realVariates(realModel.nobs * realModel.kEndog, 7L),
            realVariates(realModel.nobs * realModel.kPosdef, 11L),
            realVariates(realModel.kStates, 13L),
            workspace,
            options);
        print("simulation", name, workspace.retainedWorkspaceDoubleCount(),
            workspace.retainedResultDoubleCount(), workspace.retainedVariateDoubleCount(),
            workspace.retainedNestedScratchDoubleCount(), workspace.retainedNestedResultDoubleCount(),
            workspace.retainedTotalByteCount());
        printRealSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordComplexSimulationProfile(String name,
                                                       ModelFixture complexModel,
                                                       InitialState complexInit,
                                                       SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        SimulationSmootherEngine.simulateComplexBorrowedUnsafe(complexModel, complexInit,
            realVariates(2 * complexModel.nobs * complexModel.kEndog, 17L),
            realVariates(2 * complexModel.nobs * complexModel.kPosdef, 19L),
            realVariates(2 * complexModel.kStates, 23L),
            workspace,
            options);
        print("simulation", name, workspace.retainedComplexWorkspaceDoubleCount(),
            workspace.retainedComplexResultDoubleCount(), workspace.retainedComplexVariateDoubleCount(),
            workspace.retainedComplexNestedScratchDoubleCount(), workspace.retainedComplexNestedResultDoubleCount(),
            workspace.retainedComplexTotalByteCount());
        printComplexSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordComplexSimulationRepeatedProfile(String name,
                                                               ModelFixture complexModel,
                                                               InitialState complexInit,
                                                               SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        workspace.retainReusableWorkspaces();
        for (int i = 0; i < 2; i++) {
            SimulationSmootherEngine.simulateComplexBorrowedUnsafe(complexModel, complexInit,
                realVariates(2 * complexModel.nobs * complexModel.kEndog, 17L + i),
                realVariates(2 * complexModel.nobs * complexModel.kPosdef, 19L + i),
                realVariates(2 * complexModel.kStates, 23L + i),
                workspace,
                options);
        }
        print("simulation", name, workspace.retainedComplexWorkspaceDoubleCount(),
            workspace.retainedComplexResultDoubleCount(), workspace.retainedComplexVariateDoubleCount(),
            workspace.retainedComplexNestedScratchDoubleCount(), workspace.retainedComplexNestedResultDoubleCount(),
            workspace.retainedComplexTotalByteCount());
        printComplexSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void recordComplexSimulationRetainedCfaProfile(String name,
                                                                  ModelFixture complexModel,
                                                                  InitialState complexInit,
                                                                  SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        workspace.retainCfaPosteriorWorkspace();
        for (int i = 0; i < 2; i++) {
            SimulationSmootherEngine.simulateComplexBorrowedUnsafe(complexModel, complexInit,
                realVariates(2 * complexModel.nobs * complexModel.kEndog, 17L + i),
                realVariates(2 * complexModel.nobs * complexModel.kPosdef, 19L + i),
                realVariates(2 * complexModel.kStates, 23L + i),
                workspace,
                options);
        }
        print("simulation", name, workspace.retainedComplexWorkspaceDoubleCount(),
            workspace.retainedComplexResultDoubleCount(), workspace.retainedComplexVariateDoubleCount(),
            workspace.retainedComplexNestedScratchDoubleCount(), workspace.retainedComplexNestedResultDoubleCount(),
            workspace.retainedComplexTotalByteCount());
        printComplexSimulationWorkspaceSplit(name, workspace, options);
    }

    private static InitialState knownRealInitial(ModelFixture model) {
        return InitialState.known(new double[model.kStates], diagonal(model.kStates, 1.0, 0.1));
    }

    private static InitialState knownComplexInitial(ModelFixture model) {
        return InitialState.known(new double[2 * model.kStates], complexMatrix(diagonal(model.kStates, 1.0, 0.1)));
    }

    private static void recordComplexSimulationCopiedProfile(String name,
                                                             ModelFixture complexModel,
                                                             InitialState complexInit,
                                                             SimulationSmootherOptions options) {
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        SimulationSmootherEngine.simulateComplex(complexModel, complexInit,
            realVariates(2 * complexModel.nobs * complexModel.kEndog, 17L),
            realVariates(2 * complexModel.nobs * complexModel.kPosdef, 19L),
            realVariates(2 * complexModel.kStates, 23L),
            workspace,
            options);
        print("simulation", name, workspace.retainedComplexWorkspaceDoubleCount(),
            workspace.retainedComplexResultDoubleCount(), workspace.retainedComplexVariateDoubleCount(),
            workspace.retainedComplexNestedScratchDoubleCount(), workspace.retainedComplexNestedResultDoubleCount(),
            workspace.retainedComplexTotalByteCount());
        printComplexSimulationWorkspaceSplit(name, workspace, options);
    }

    private static void printRealSimulationWorkspaceSplit(String name,
                                                          SimulationSmootherEngine.Workspace workspace,
                                                          SimulationSmootherOptions options) {
        if (options.method() != SimulationSmootherOptions.Method.CFA) {
            return;
        }
        long cfa = workspace.retainedCfaWorkspaceDoubleCount();
        long combine = workspace.retainedCombineWorkspaceDoubleCount();
        print("simulation-cfa", name + "-cfa-workspace", cfa, 0, 0, 0, 0, cfa * Double.BYTES);
        print("simulation-cfa", name + "-combine-workspace", combine, 0, 0, 0, 0, combine * Double.BYTES);
    }

    private static void printComplexSimulationWorkspaceSplit(String name,
                                                             SimulationSmootherEngine.Workspace workspace,
                                                             SimulationSmootherOptions options) {
        if (options.method() != SimulationSmootherOptions.Method.CFA) {
            return;
        }
        long cfa = workspace.retainedComplexCfaWorkspaceDoubleCount();
        long combine = workspace.retainedComplexCombineWorkspaceDoubleCount();
        print("simulation-cfa", name + "-cfa-workspace", cfa, 0, 0, 0, 0, cfa * Double.BYTES);
        print("simulation-cfa", name + "-combine-workspace", combine, 0, 0, 0, 0, combine * Double.BYTES);
    }

    private static ModelFixture buildRealModel(int nobs) {
        ModelFixture model = new ModelFixture(4, 4, 4, nobs);
        double[] design = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        double[] transition = {
            0.85, 0.05, 0.00, 0.00,
            0.02, 0.80, 0.03, 0.00,
            0.00, 0.04, 0.78, 0.06,
            0.00, 0.00, 0.02, 0.75
        };
        double[] selection = identity(4);
        double[] obsCov = diagonal(4, 0.4, 0.1);
        double[] stateCov = diagonal(4, 0.2, 0.0);
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[4], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[4], t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{0.8 + 0.04 * t, -0.3 + 0.02 * t, 0.2 - 0.01 * t, 0.1 + 0.03 * t}, t);
        }
        return model;
    }

    private static ModelFixture buildCollapsedEligibleModel(int nobs) {
        ModelFixture model = new ModelFixture(3, 2, 2, nobs);
        for (int t = 0; t < nobs; t++) {
            model.setDesign(new double[]{0.5, 0.2, 0.0, 0.8, 1.0, -0.5}, t);
            model.setObsIntercept(new double[3], t);
            model.setObsCov(new double[]{0.2, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.5}, t);
            model.setTransition(new double[]{0.4, 0.5, 1.0, 0.0}, t);
            model.setStateIntercept(new double[]{0.03, -0.01}, t);
            model.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            model.setStateCov(new double[]{2.0, 0.0, 0.0, 1.0}, t);
            model.setEndog(new double[]{1.0 + 0.2 * t, -0.3 + 0.1 * t, 0.6 - 0.15 * t}, t);
        }
        return model;
    }

    private static ModelFixture buildWideCollapsedEligibleModel(int nobs) {
        ModelFixture model = new ModelFixture(8, 2, 2, nobs);
        double[] design = {
            0.5, 0.2,
            0.0, 0.8,
            1.0, -0.5,
            -0.3, 0.7,
            0.6, 0.1,
            -0.2, 0.4,
            0.9, 0.3,
            0.1, -0.6
        };
        double[] obsCov = new double[64];
        for (int row = 0; row < 8; row++) {
            obsCov[row * 8 + row] = 0.4 + 0.1 * row;
        }
        double[] transition = {0.4, 0.5, 1.0, 0.0};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {2.0, 0.0, 0.0, 1.0};
        for (int t = 0; t < nobs; t++) {
            double[] endog = new double[8];
            for (int obs = 0; obs < 8; obs++) {
                endog[obs] = 0.4 + 0.05 * obs + 0.02 * t * (obs + 1);
            }
            model.setDesign(design, t);
            model.setObsIntercept(new double[8], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[]{0.03, -0.01}, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(endog, t);
        }
        return model;
    }

    private static ModelFixture buildRealObservationTransformModel(int nobs) {
        ModelFixture model = new ModelFixture(3, 2, 2, nobs);
        double[] design = {1.0, 0.2, 0.1, 1.0, -0.3, 0.7};
        double[] transition = {0.65, 0.08, 0.02, 0.7};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] obsCov = {1.0, 0.18, 0.08, 0.18, 1.2, 0.12, 0.08, 0.12, 0.9};
        double[] stateCov = {0.25, 0.0, 0.0, 0.2};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[3], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[]{0.02, -0.01}, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{0.7 + 0.03 * t, -0.2 + 0.02 * t, 0.1 - 0.01 * t}, t);
        }
        return model;
    }

    private static KalmanSSM buildChandrasekharModel(int nobs) {
        double[] endog = new double[2 * nobs];
        for (int t = 0; t < nobs; t++) {
            endog[2 * t] = 0.25 + 0.04 * t;
            endog[2 * t + 1] = -0.5 + 0.03 * t;
        }
        return KalmanSSM.builder(2, 2, 2, nobs)
            .design(new double[]{1.0, 0.0, 0.0, 1.0}, false)
            .obsIntercept(new double[2], false)
            .obsCov(new double[]{1.0, 0.0, 0.0, 1.2}, false)
            .transition(new double[]{0.7, 0.1, 0.0, 0.65}, false)
            .stateIntercept(new double[2], false)
            .selection(new double[]{1.0, 0.0, 0.0, 1.0}, false)
            .stateCovariance(new double[]{0.5, 0.0, 0.0, 0.4}, false)
            .endog(endog)
            .allObserved()
            .build();
    }

    private static ModelFixture buildComplexModel(int nobs) {
        ModelFixture model = ModelFixture.complex(2, 2, 2, nobs);
        double[] design = complexMatrix(new double[]{1.0, 0.1, 0.2, 1.0});
        double[] obsCov = complexMatrix(new double[]{0.6, 0.0, 0.0, 0.8});
        double[] transition = complexMatrix(new double[]{0.7, 0.1, 0.0, 0.6});
        double[] selection = complexMatrix(new double[]{1.0, 0.0, 0.0, 1.0});
        double[] stateCov = complexMatrix(new double[]{0.2, 0.0, 0.0, 0.25});
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[4], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[4], t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{1.0 + 0.2 * t, 0.05 * t, -0.4 + 0.1 * t, -0.02 * t}, t);
        }
        return model;
    }

    private static ModelFixture buildComplexObservationTransformModel(int nobs) {
        ModelFixture model = ModelFixture.complex(2, 2, 2, nobs);
        double[] design = complexMatrix(new double[]{1.0, 0.2, -0.1, 0.9});
        double[] obsCov = complexMatrix(new double[]{0.9, 0.14, 0.14, 0.7});
        double[] transition = complexMatrix(new double[]{0.68, 0.06, 0.02, 0.61});
        double[] selection = complexMatrix(new double[]{1.0, 0.0, 0.0, 1.0});
        double[] stateCov = complexMatrix(new double[]{0.18, 0.0, 0.0, 0.22});
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[4], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[4], t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{0.8 + 0.04 * t, 0.01 * t, -0.3 + 0.02 * t, -0.015 * t}, t);
        }
        return model;
    }

    private static double[] identity(int dimension) {
        double[] values = new double[dimension * dimension];
        for (int i = 0; i < dimension; i++) {
            values[i * dimension + i] = 1.0;
        }
        return values;
    }

    private static double[] diagonal(int dimension, double start, double step) {
        double[] values = new double[dimension * dimension];
        for (int i = 0; i < dimension; i++) {
            values[i * dimension + i] = start + step * i;
        }
        return values;
    }

    private static double[] complexMatrix(double[] realValues) {
        double[] values = new double[2 * realValues.length];
        for (int i = 0; i < realValues.length; i++) {
            values[2 * i] = realValues[i];
        }
        return values;
    }

    private static double[] realVariates(int length, long seed) {
        double[] values = new double[length];
        Random rng = new Random(seed);
        for (int i = 0; i < values.length; i++) {
            values[i] = rng.nextGaussian();
        }
        return values;
    }

    private static long diffuseSurfaceDoubles(FilterResult result) {
        return (long) result.predictedDiffuseStateCovLength() + result.forecastErrorDiffuseCovLength();
    }

    private static long diffuseSurfaceDoubles(ZFilterResult result) {
        return (long) result.predictedDiffuseStateCovLength() + result.forecastErrorDiffuseCovLength();
    }

    private static void recordScratchLayouts(ModelFixture realModel, ModelFixture complexModel) {
        int rkEndog = realModel.kEndog;
        int rkStates = realModel.kStates;
        int rkPosdef = realModel.kPosdef;
        int zkEndog = complexModel.kEndog;
        int zkStates = complexModel.kStates;
        int zkPosdef = complexModel.kPosdef;

        recordFilterLayout("real-conventional", FilterScratchLayout.create(1, rkEndog, rkStates, rkPosdef, false, false, false),
            1, rkEndog, rkStates, rkPosdef,
            conventionalOptimizedLength(1, rkEndog, rkStates, rkPosdef));
        recordUnivariateLayout("real-univariate", UnivariateScratchLayout.createRegular(rkEndog, rkStates, rkPosdef, false, false),
            rkStates, rkPosdef,
            univariateRegularOptimizedLength(rkStates, rkPosdef));
        recordUnivariateLayout("real-univariate-exact-diffuse", UnivariateScratchLayout.createExactDiffuse(rkEndog, rkStates, rkPosdef, false, false, false),
            rkStates, rkPosdef,
            univariateExactDiffuseOptimizedLength(rkStates, rkPosdef));
        recordZFilterLayout("complex-conventional", ZFilterScratchLayout.createConventional(zkEndog, zkStates, zkPosdef, false, false, false),
            zkEndog, zkStates, zkPosdef,
            zConventionalOptimizedLength(zkEndog, zkStates, zkPosdef));
        recordZFilterLayout("complex-univariate", ZFilterScratchLayout.createUnivariate(zkEndog, zkStates, zkPosdef, false, false),
            zkEndog, zkStates, zkPosdef,
            zUnivariateOptimizedLength(zkStates, zkPosdef));
        recordZFilterLayout("complex-univariate-exact-diffuse", ZFilterScratchLayout.createExactDiffuse(zkEndog, zkStates, zkPosdef, false, false, false),
            zkEndog, zkStates, zkPosdef,
            zExactDiffuseOptimizedLength(zkStates, zkPosdef));
        recordSmootherLayout("real-smoother-full", SmootherScratchLayout.create(rkEndog, rkStates,
            true, true, true, true, true, true, false), rkEndog, rkStates, -1);
        recordZSmootherLayout("complex-smoother-full", ZSmootherScratchLayout.create(zkEndog, zkStates, zkPosdef,
            true, true, true, true, true, true, false), zkEndog, zkStates, zkPosdef, -1);
    }

    private static int conventionalOptimizedLength(int scalarWidth, int kEndog, int kStates, int kPosdef) {
        int ksKe = scalarWidth * kStates * kEndog;
        int keKe = scalarWidth * kEndog * kEndog;
        int updateLength = ksKe + ksKe + keKe + scalarWidth * kEndog;
        int predictionLength = Math.max(scalarWidth * kStates * kStates, scalarWidth * kStates * kPosdef);
        return Math.max(updateLength, predictionLength);
    }

    private static int univariateRegularOptimizedLength(int kStates, int kPosdef) {
        return kStates + kStates * kStates + Math.max(kStates, Math.max(kStates * kStates, kStates * kPosdef));
    }

    private static int univariateExactDiffuseOptimizedLength(int kStates, int kPosdef) {
        return kStates + 2 * kStates * kStates + Math.max(4 * kStates, Math.max(kStates * kStates, kStates * kPosdef));
    }

    private static int zConventionalOptimizedLength(int kEndog, int kStates, int kPosdef) {
        return conventionalOptimizedLength(2, kEndog, kStates, kPosdef);
    }

    private static int zUnivariateOptimizedLength(int kStates, int kPosdef) {
        return 2 * kStates + 2 * kStates * kStates
            + Math.max(2 * kStates, Math.max(2 * kStates * kStates, 2 * kStates * kPosdef));
    }

    private static int zExactDiffuseOptimizedLength(int kStates, int kPosdef) {
        return 2 * kStates + 4 * kStates * kStates
            + Math.max(8 * kStates, Math.max(2 * kStates * kStates, 2 * kStates * kPosdef));
    }

    private static void recordFilterLayout(String name,
                                           FilterScratchLayout layout,
                                           int scalarWidth,
                                           int kEndog,
                                           int kStates,
                                           int kPosdef,
                                           int expectedOptimizedLength) {
        printLayoutSlice(name, "tmpK", layout.tmpKBase(), scalarWidth * kStates * kEndog, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPZt", layout.tmpPZtBase(), scalarWidth * kStates * kEndog, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpFcopy", layout.tmpFcopyBase(), scalarWidth * kEndog * kEndog, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpTP", layout.tmpTPBase(), scalarWidth * kStates * kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpV", layout.tmpVBase(), scalarWidth * kEndog, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpRQ", layout.tmpRQBase(), scalarWidth * kStates * kPosdef, true, layout.totalLength(), expectedOptimizedLength);
    }

    private static void recordUnivariateLayout(String name,
                                               UnivariateScratchLayout layout,
                                               int kStates,
                                               int kPosdef,
                                               int expectedOptimizedLength) {
        int ksKs = kStates * kStates;
        printLayoutSlice(name, "tmpA", layout.tmpABase(), kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpP", layout.tmpPBase(), activeLength(layout.tmpPBase(), ksKs), layout.tmpPBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPZi", layout.tmpPZiBase(), activeLength(layout.tmpPZiBase(), kStates), layout.tmpPZiBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpRQ", layout.tmpRQBase(), activeLength(layout.tmpRQBase(), kStates * kPosdef), layout.tmpRQBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPInf", layout.tmpPInfBase(), activeLength(layout.tmpPInfBase(), ksKs), layout.tmpPInfBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPStar", layout.tmpPStarBase(), activeLength(layout.tmpPStarBase(), ksKs), layout.tmpPStarBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpTP", layout.tmpTPBase(), activeLength(layout.tmpTPBase(), ksKs), layout.tmpTPBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpMStar", layout.tmpMStarBase(), activeLength(layout.tmpMStarBase(), kStates), layout.tmpMStarBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpMInf", layout.tmpMInfBase(), activeLength(layout.tmpMInfBase(), kStates), layout.tmpMInfBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpK0", layout.tmpK0Base(), activeLength(layout.tmpK0Base(), kStates), layout.tmpK0Base() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpK1", layout.tmpK1Base(), activeLength(layout.tmpK1Base(), kStates), layout.tmpK1Base() >= 0, layout.totalLength(), expectedOptimizedLength);
    }

    private static void recordZFilterLayout(String name,
                                            ZFilterScratchLayout layout,
                                            int kEndog,
                                            int kStates,
                                            int kPosdef,
                                            int expectedOptimizedLength) {
        int ksKe = 2 * kStates * kEndog;
        int ksKs = 2 * kStates * kStates;
        printLayoutSlice(name, "tmpA", layout.tmpABase(), activeLength(layout.tmpABase(), 2 * kStates), layout.tmpABase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpK", layout.tmpKBase(), activeLength(layout.tmpKBase(), ksKe), layout.tmpKBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPZt", layout.tmpPZtBase(), activeLength(layout.tmpPZtBase(), ksKe), layout.tmpPZtBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpFcopy", layout.tmpFcopyBase(), activeLength(layout.tmpFcopyBase(), 2 * kEndog * kEndog), layout.tmpFcopyBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpTP", layout.tmpTPBase(), activeLength(layout.tmpTPBase(), ksKs), layout.tmpTPBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpV", layout.tmpVBase(), activeLength(layout.tmpVBase(), 2 * kEndog), layout.tmpVBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpRQ", layout.tmpRQBase(), activeLength(layout.tmpRQBase(), 2 * kStates * kPosdef), layout.tmpRQBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPInf", layout.tmpPInfBase(), activeLength(layout.tmpPInfBase(), ksKs), layout.tmpPInfBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpPStar", layout.tmpPStarBase(), activeLength(layout.tmpPStarBase(), ksKs), layout.tmpPStarBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpMStar", layout.tmpMStarBase(), activeLength(layout.tmpMStarBase(), 2 * kStates), layout.tmpMStarBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpMInf", layout.tmpMInfBase(), activeLength(layout.tmpMInfBase(), 2 * kStates), layout.tmpMInfBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpK0", layout.tmpK0Base(), activeLength(layout.tmpK0Base(), 2 * kStates), layout.tmpK0Base() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpK1", layout.tmpK1Base(), activeLength(layout.tmpK1Base(), 2 * kStates), layout.tmpK1Base() >= 0, layout.totalLength(), expectedOptimizedLength);
    }

    private static void recordSmootherLayout(String name, SmootherScratchLayout layout, int kEndog, int kStates, int expectedOptimizedLength) {
        printLayoutSlice(name, "r", layout.rBase(), kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "n", layout.nBase(), kStates * kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpRq", layout.tmpRqBase(), activeLength(layout.tmpRqBase(), kStates * kStates), layout.tmpRqBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpFcopy", layout.tmpFcopyBase(), activeLength(layout.tmpFcopyBase(), kEndog * kEndog), layout.tmpFcopyBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpTk", layout.tmpTkBase(), activeLength(layout.tmpTkBase(), kStates * kEndog), layout.tmpTkBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpKh", layout.tmpKhBase(), activeLength(layout.tmpKhBase(), kStates * kEndog), layout.tmpKhBase() >= 0, layout.totalLength(), expectedOptimizedLength);
    }

    private static void recordZSmootherLayout(String name,
                                              ZSmootherScratchLayout layout,
                                              int kEndog,
                                              int kStates,
                                              int kPosdef,
                                              int expectedOptimizedLength) {
        printLayoutSlice(name, "r", layout.rBase(), 2 * kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "n", layout.nBase(), 2 * kStates * kStates, true, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpRq", layout.tmpRqBase(), activeLength(layout.tmpRqBase(), Math.max(2 * kStates * kStates, 2 * kStates * kPosdef)), layout.tmpRqBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpFcopy", layout.tmpFcopyBase(), activeLength(layout.tmpFcopyBase(), 2 * kEndog * kEndog), layout.tmpFcopyBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpTk", layout.tmpTkBase(), activeLength(layout.tmpTkBase(), 2 * kStates * kEndog), layout.tmpTkBase() >= 0, layout.totalLength(), expectedOptimizedLength);
        printLayoutSlice(name, "tmpKh", layout.tmpKhBase(), activeLength(layout.tmpKhBase(), 2 * kStates * kEndog), layout.tmpKhBase() >= 0, layout.totalLength(), expectedOptimizedLength);
    }

    private static int activeLength(int base, int length) {
        return base < 0 ? 0 : length;
    }

    private static void printLayoutSlice(String name,
                                         String slice,
                                         int base,
                                         int length,
                                         boolean active,
                                         int totalLength,
                                         int expectedOptimizedLength) {
        System.out.printf("layout,%s,%s,%d,%d,%s,%d,%d%n",
            name, slice, base, length, active, totalLength, expectedOptimizedLength);
    }

    private static void print(String category,
                              String name,
                              long scratch,
                              long result,
                              long diffuseOrVariates,
                              long nestedScratch,
                              long nestedResult,
                              long totalBytes) {
        System.out.printf("%s,%s,%d,%d,%d,%d,%d,%d%n",
            category, name, scratch, result, diffuseOrVariates, nestedScratch, nestedResult, totalBytes);
    }
}
