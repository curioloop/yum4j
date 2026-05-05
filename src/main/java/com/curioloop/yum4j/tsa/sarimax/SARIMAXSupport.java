package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.reg.OLS;

import java.util.ArrayList;
import java.util.Arrays;

final class SARIMAXSupport {

    private SARIMAXSupport() {
    }

    static Meta analyze(SARIMAXSpec spec) {
        double[] endog = spec.endog();
        double[][] exog = spec.exog();
        int[] trendPowers = spec.trendPowers();
        ARIMAOrder order = spec.order();
        SeasonalOrder seasonalOrder = spec.seasonalOrder();
        int seasonalPeriod = seasonalOrder.period();
        if (spec.simpleDifferencing() && (order.integration() > 0 || seasonalOrder.integration() > 0)) {
            DifferencedData differenced = differenceData(
                endog,
                exog,
                order.integration(),
                seasonalOrder.integration(),
                seasonalPeriod);
            endog = differenced.endog();
            exog = differenced.exog();
        }
        int nobs = endog.length;
        int kExog = exog == null || exog.length == 0 ? 0 : exog[0].length;
        boolean stateRegression = !spec.mleRegression() && kExog > 0;
        boolean timeVaryingRegression = stateRegression && spec.timeVaryingRegression();
        int kTrend = trendPowers.length;
        int stateDiff = spec.simpleDifferencing() ? 0 : order.integration();
        int stateSeasonalDiff = spec.simpleDifferencing() ? 0 : seasonalOrder.integration();
        int reducedArOrder = order.autoregressive() + seasonalOrder.autoregressive() * seasonalPeriod;
        int reducedMaOrder = order.movingAverage() + seasonalOrder.movingAverage() * seasonalPeriod;
        int totalDiff = stateDiff + stateSeasonalDiff * seasonalPeriod;
        int totalOrder = Math.max(reducedArOrder, reducedMaOrder + 1);
        int kStates = totalDiff + totalOrder + (stateRegression ? kExog : 0);
        int kPosdef = (totalOrder > 0 ? 1 : 0) + (timeVaryingRegression ? kExog : 0);

        double[][] trendData = buildTrendData(trendPowers, nobs);
        String[] parameterNames = buildParameterNames(
            trendPowers,
            stateRegression ? 0 : kExog,
            spec.autoregressiveLags(),
            spec.movingAverageLags(),
            spec.seasonalAutoregressiveLags(),
            spec.seasonalMovingAverageLags(),
            spec.measurementError(),
            spec.concentrateScale(),
            timeVaryingRegression ? kExog : 0);

        Meta draft = new Meta(
            spec,
            endog,
            exog,
            trendPowers,
            trendData,
            nobs,
            kExog,
            kTrend,
            reducedArOrder,
            reducedMaOrder,
            totalDiff,
            totalOrder,
            kStates,
            kPosdef,
            stateRegression,
            timeVaryingRegression,
            parameterNames,
            null,
            null);
        StateSpaceModel templateModel = buildTemplateModel(draft);
        double[] startParams = initialParams(draft);
        return new Meta(
            spec,
            endog,
            exog,
            trendPowers,
            trendData,
            nobs,
            kExog,
            kTrend,
            reducedArOrder,
            reducedMaOrder,
            totalDiff,
            totalOrder,
            kStates,
            kPosdef,
            stateRegression,
            timeVaryingRegression,
            parameterNames,
            startParams,
            templateModel);
    }

    static Meta extend(Meta base, SARIMAXSpec spec) {
        double[] endog = spec.endog();
        double[][] exog = spec.exog();
        ARIMAOrder order = spec.order();
        SeasonalOrder seasonalOrder = spec.seasonalOrder();
        int seasonalPeriod = seasonalOrder.period();
        if (spec.simpleDifferencing() && (order.integration() > 0 || seasonalOrder.integration() > 0)) {
            DifferencedData differenced = differenceData(
                endog,
                exog,
                order.integration(),
                seasonalOrder.integration(),
                seasonalPeriod);
            endog = differenced.endog();
            exog = differenced.exog();
        }
        int nobs = endog.length;
        int kExog = exog == null || exog.length == 0 ? 0 : exog[0].length;
        if (kExog != base.kExog()) {
            throw new IllegalArgumentException("extended exog column count must match the original model");
        }
        double[][] trendData = buildTrendData(base.trendPowers(), nobs);
        Meta draft = new Meta(
            spec,
            endog,
            exog,
            base.trendPowers(),
            trendData,
            nobs,
            base.kExog(),
            base.kTrend(),
            base.reducedArOrder(),
            base.reducedMaOrder(),
            base.totalDiff(),
            base.totalOrder(),
            base.kStates(),
            base.kPosdef(),
            base.stateRegression(),
            base.timeVaryingRegression(),
            base.parameterNames(),
            base.startParams(),
            null);
        StateSpaceModel templateModel = buildTemplateModel(draft);
        return new Meta(
            spec,
            endog,
            exog,
            base.trendPowers(),
            trendData,
            nobs,
            base.kExog(),
            base.kTrend(),
            base.reducedArOrder(),
            base.reducedMaOrder(),
            base.totalDiff(),
            base.totalOrder(),
            base.kStates(),
            base.kPosdef(),
            base.stateRegression(),
            base.timeVaryingRegression(),
            base.parameterNames(),
            base.startParams(),
            templateModel);
    }

    static double[] initialParams(Meta meta) {
        PreparedData prepared = prepareData(meta);
        double[] endogenous = prepared.endog();
        double[][] exogenous = prepared.exog();
        double[][] trend = prepared.trend();

        double[] exogParams = meta.stateRegression() ? new double[0] : new double[meta.kExog()];
        double[] workingEndog = Arrays.copyOf(endogenous, endogenous.length);
        if (meta.kExog() > 0 && workingEndog.length > 0) {
            OLSFit regression = fitOLS(workingEndog, exogenous);
            if (!meta.stateRegression()) {
                exogParams = Arrays.copyOf(regression.params(), regression.params().length);
            }
            workingEndog = regression.residuals();
        }

        CSSStart nonSeasonal = condSumSquare(
            meta.spec().autoregressiveLags(),
            meta.spec().movingAverageLags(),
            workingEndog,
            trend);
        CSSStart seasonal = condSumSquare(
            meta.spec().seasonalAutoregressiveLags(),
            meta.spec().seasonalMovingAverageLags(),
            workingEndog,
            null);

        double[] arParams = stabilizeInitialBlock(nonSeasonal.arParams(), true);
        double[] maParams = stabilizeInitialBlock(nonSeasonal.maParams(), false);
        double[] seasonalArParams = stabilizeInitialBlock(seasonal.arParams(), true);
        double[] seasonalMaParams = stabilizeInitialBlock(seasonal.maParams(), false);

        double variance = !Double.isNaN(nonSeasonal.variance())
            ? nonSeasonal.variance()
            : seasonal.variance();
        if (Double.isNaN(variance)) {
            variance = variance(workingEndog);
        }
        variance = Math.max(variance, 1e-8);

        return concat(
            nonSeasonal.trendParams(),
            exogParams,
            arParams,
            maParams,
            seasonalArParams,
            seasonalMaParams,
            meta.timeVaryingRegression() ? fillOnes(meta.kExog()) : new double[0],
            meta.spec().measurementError() ? new double[]{1.0} : new double[0],
            meta.spec().concentrateScale() ? new double[0] : new double[]{variance});
    }

    static PreparedData prepareData(Meta meta) {
        if (meta.spec().simpleDifferencing()) {
            return dropMissing(meta.endog(), meta.exog(), meta.trendData());
        }
        double[] endogenous = Arrays.copyOf(meta.endog(), meta.endog().length);
        double[][] exogenous = copy2d(meta.exog());
        double[][] trend = copy2d(meta.trendData());

        SeasonalOrder seasonalOrder = meta.spec().seasonalOrder();
        for (int diff = seasonalOrder.integration(); diff > 0; diff--) {
            int period = seasonalOrder.period();
            for (int index = endogenous.length - 1 - period; index >= 0; index--) {
                endogenous[index + period] -= endogenous[index];
            }
            if (exogenous != null) {
                for (int index = exogenous.length - 1 - period; index >= 0; index--) {
                    for (int col = 0; col < exogenous[index].length; col++) {
                        exogenous[index + period][col] -= exogenous[index][col];
                    }
                }
            }
            endogenous = Arrays.copyOfRange(endogenous, period, endogenous.length);
            if (exogenous != null) {
                exogenous = Arrays.copyOfRange(exogenous, period, exogenous.length);
            }
        }

        for (int diff = meta.spec().order().integration(); diff > 0; diff--) {
            for (int index = endogenous.length - 2; index >= 0; index--) {
                endogenous[index + 1] -= endogenous[index];
            }
            if (exogenous != null) {
                for (int index = exogenous.length - 2; index >= 0; index--) {
                    for (int col = 0; col < exogenous[index].length; col++) {
                        exogenous[index + 1][col] -= exogenous[index][col];
                    }
                }
            }
            endogenous = Arrays.copyOfRange(endogenous, 1, endogenous.length);
            if (exogenous != null) {
                exogenous = Arrays.copyOfRange(exogenous, 1, exogenous.length);
            }
        }

        if (trend != null) {
            trend = Arrays.copyOf(trend, endogenous.length);
        }

        return dropMissing(endogenous, exogenous, trend);
    }

    static double[] constrainStationary(double sign, double[] unconstrained) {
        int n = unconstrained.length;
        double[] params = new double[n];
        constrainStationary(sign, unconstrained, 0, n, params, 0, new StationaryTransformWorkspace());
        return params;
    }

    static void constrainStationary(double sign,
                                    double[] unconstrained,
                                    int sourceOffset,
                                    int length,
                                    double[] target,
                                    int targetOffset,
                                    StationaryTransformWorkspace workspace) {
        int n = length;
        if (n == 0) {
            return;
        }
        workspace.ensure(n);
        double[] reflection = workspace.reflection;
        double[] pacf = workspace.matrix;
        for (int i = 0; i < n; i++) {
            double u = unconstrained[sourceOffset + i];
            reflection[i] = u / Math.sqrt(1.0 + u * u);
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < k; i++) {
                pacf[k * n + i] = pacf[(k - 1) * n + i] + reflection[k] * pacf[(k - 1) * n + k - 1 - i];
            }
            pacf[k * n + k] = reflection[k];
        }
        for (int i = 0; i < n; i++) {
            target[targetOffset + i] = -pacf[(n - 1) * n + i] * sign;
        }
    }

    static double[] unconstrainStationary(double sign, double[] constrained) {
        int n = constrained.length;
        double[] result = new double[n];
        unconstrainStationary(sign, constrained, 0, n, result, 0, new StationaryTransformWorkspace());
        return result;
    }

    static void unconstrainStationary(double sign,
                                      double[] constrained,
                                      int sourceOffset,
                                      int length,
                                      double[] target,
                                      int targetOffset,
                                      StationaryTransformWorkspace workspace) {
        int n = length;
        if (n == 0) {
            return;
        }
        workspace.ensure(n);
        double[] matrix = workspace.matrix;
        int lastRowBase = (n - 1) * n;
        for (int i = 0; i < n; i++) {
            matrix[lastRowBase + i] = -(constrained[sourceOffset + i] * sign);
        }
        for (int k = n - 1; k > 0; k--) {
            double kk = matrix[k * n + k];
            double denom = 1.0 - kk * kk;
            if (!(denom > 0.0)) {
                Arrays.fill(target, targetOffset, targetOffset + n, 0.0);
                return;
            }
            for (int i = 0; i < k; i++) {
                matrix[(k - 1) * n + i] = (matrix[k * n + i] - kk * matrix[k * n + k - i - 1]) / denom;
            }
        }
        for (int i = 0; i < n; i++) {
            double r = matrix[i * n + i];
            double denom = 1.0 - r * r;
            if (!(denom > 0.0)) {
                Arrays.fill(target, targetOffset, targetOffset + n, 0.0);
                return;
            }
            target[targetOffset + i] = r / Math.sqrt(denom);
        }
    }

    static StateSpaceModel buildTemplateModel(Meta meta) {
        double[] endog = Arrays.copyOf(meta.endog(), meta.endog().length);
        boolean[] missing = new boolean[endog.length];
        int[] nmissing = new int[meta.nobs()];
        for (int t = 0; t < meta.nobs(); t++) {
            if (Double.isNaN(endog[t])) {
                missing[t] = true;
                nmissing[t] = 1;
            }
        }
        return StateSpaceModel.builder(1, meta.kStates(), meta.kPosdef(), meta.nobs())
            .design(initDesign(meta), meta.stateRegression())
            .obsIntercept((!meta.stateRegression() && meta.kExog() > 0) || (meta.kTrend() > 0 && meta.spec().hamiltonRepresentation())
                ? new double[meta.nobs()]
                : new double[1],
                (!meta.stateRegression() && meta.kExog() > 0) || (meta.kTrend() > 0 && meta.spec().hamiltonRepresentation()))
            .obsCov(new double[1], false)
            .transition(initTransition(meta), false)
            .stateIntercept(meta.kTrend() > 0 ? new double[meta.nobs() * meta.kStates()] : new double[meta.kStates()], meta.kTrend() > 0)
            .selection(initSelection(meta), false)
            .stateCovariance(new double[Math.max(1, meta.kPosdef() * meta.kPosdef())], false)
            .endog(endog)
            .missing(missing, nmissing)
            .build();
    }

    static double[] initDesign(Meta meta) {
        ArrayList<Double> design = new ArrayList<>();
        for (int i = 0; i < stateDiff(meta); i++) {
            design.add(1.0);
        }
        SeasonalOrder seasonalOrder = meta.spec().seasonalOrder();
        for (int block = 0; block < stateSeasonalDiff(meta); block++) {
            for (int i = 0; i < seasonalOrder.period() - 1; i++) {
                design.add(0.0);
            }
            design.add(1.0);
        }
        if (meta.kPosdef() > 0) {
            design.add(1.0);
        }
        for (int i = 1; i < meta.totalOrder(); i++) {
            design.add(0.0);
        }
        if (design.isEmpty()) {
            design.add(0.0);
        }
        double[] baseDesign = new double[meta.kStates()];
        double[] staticDesign = toArray(design);
        System.arraycopy(staticDesign, 0, baseDesign, 0, staticDesign.length);
        if (!meta.stateRegression()) {
            return baseDesign;
        }

        double[] timeVarying = new double[meta.nobs() * meta.kStates()];
        int exogStart = exogStateStart(meta);
        for (int t = 0; t < meta.nobs(); t++) {
            int offset = t * meta.kStates();
            System.arraycopy(baseDesign, 0, timeVarying, offset, baseDesign.length);
            System.arraycopy(meta.exog()[t], 0, timeVarying, offset + exogStart, meta.kExog());
        }
        return timeVarying;
    }

    static double[] initTransition(Meta meta) {
        double[] transition = new double[meta.kStates() * meta.kStates()];
        int start = meta.totalDiff();

        if (meta.totalOrder() > 0) {
            double[] companion = companionMatrix(meta.totalOrder());
            for (int row = 0; row < meta.totalOrder(); row++) {
                for (int col = 0; col < meta.totalOrder(); col++) {
                    int source = meta.spec().hamiltonRepresentation()
                        ? col * meta.totalOrder() + row
                        : row * meta.totalOrder() + col;
                    transition[(start + row) * meta.kStates() + start + col] = companion[source];
                }
            }
        }

        if (meta.stateRegression()) {
            int exogStart = exogStateStart(meta);
            for (int index = 0; index < meta.kExog(); index++) {
                transition[(exogStart + index) * meta.kStates() + exogStart + index] = 1.0;
            }
        }

        SeasonalOrder seasonalOrder = meta.spec().seasonalOrder();
        if (stateSeasonalDiff(meta) > 0) {
            int period = seasonalOrder.period();
            double[] companion = companionMatrix(period);
            companion[(period - 1) * period] = 1.0;
            for (int block = 0; block < stateSeasonalDiff(meta); block++) {
                start = stateDiff(meta) + block * period;
                for (int row = 0; row < period; row++) {
                    for (int col = 0; col < period; col++) {
                        transition[(start + row) * meta.kStates() + start + col] = companion[col * period + row];
                    }
                }
                if (block < stateSeasonalDiff(meta) - 1) {
                    int next = stateDiff(meta) + (block + 1) * period;
                    transition[start * meta.kStates() + next + period - 1] = 1.0;
                }
                transition[start * meta.kStates() + meta.totalDiff()] = 1.0;
            }
        }

        if (stateDiff(meta) > 0) {
            int d = stateDiff(meta);
            for (int row = 0; row < d; row++) {
                for (int col = row; col < d; col++) {
                    transition[row * meta.kStates() + col] = 1.0;
                }
                transition[row * meta.kStates() + meta.totalDiff()] = 1.0;
            }
            if (seasonalOrder.period() > 0) {
                double[] seasonalOnes = new double[seasonalOrder.period() * stateSeasonalDiff(meta)];
                for (int block = 0; block < stateSeasonalDiff(meta); block++) {
                    seasonalOnes[block * seasonalOrder.period() + seasonalOrder.period() - 1] = 1.0;
                }
                int begin = d;
                int end = meta.totalDiff();
                for (int row = 0; row < d; row++) {
                    System.arraycopy(seasonalOnes, 0, transition, row * meta.kStates() + begin, end - begin);
                }
            }
        }
        return transition;
    }

    static double[] initSelection(Meta meta) {
        double[] selection = new double[meta.kStates() * Math.max(1, meta.kPosdef())];
        if (meta.kPosdef() == 0) {
            return selection;
        }
        if (meta.totalOrder() > 0) {
            selection[meta.totalDiff() * meta.kPosdef()] = 1.0;
        }
        if (meta.timeVaryingRegression()) {
            int exogStart = exogStateStart(meta);
            int columnOffset = meta.totalOrder() > 0 ? 1 : 0;
            for (int index = 0; index < meta.kExog(); index++) {
                selection[(exogStart + index) * meta.kPosdef() + columnOffset + index] = 1.0;
            }
        }
        return selection;
    }

    static double[] companionMatrix(int order) {
        double[] companion = new double[order * order];
        for (int row = 0; row < order - 1; row++) {
            companion[row * order + row + 1] = 1.0;
        }
        return companion;
    }

    static double[] multiply(double[] left, double[] right) {
        double[] result = new double[left.length + right.length - 1];
        for (int i = 0; i < left.length; i++) {
            for (int j = 0; j < right.length; j++) {
                result[i + j] += left[i] * right[j];
            }
        }
        return result;
    }

    static double[] lagPolynomial(double[] params, int[] lags, boolean autoregressive) {
        int maxLag = lastLag(lags);
        double[] polynomial = new double[maxLag + 1];
        polynomial[0] = 1.0;
        for (int i = 0; i < params.length; i++) {
            polynomial[lags[i]] = autoregressive ? -params[i] : params[i];
        }
        return polynomial;
    }

    static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static double[][] buildTrendData(int[] trendPowers, int nobs) {
        if (trendPowers.length == 0) {
            return null;
        }
        double[][] trendData = new double[nobs][trendPowers.length];
        for (int row = 0; row < nobs; row++) {
            int t = row + 1;
            for (int col = 0; col < trendPowers.length; col++) {
                trendData[row][col] = Math.pow(t, trendPowers[col]);
            }
        }
        return trendData;
    }

    private static String[] buildParameterNames(int[] trendPowers,
                                                int kExog,
                                                int[] arLags,
                                                int[] maLags,
                                                int[] seasonalArLags,
                                                int[] seasonalMaLags,
                                                boolean measurementError,
                                                boolean concentrateScale,
                                                int exogVarianceCount) {
        ArrayList<String> names = new ArrayList<>();
        for (int power : trendPowers) {
            names.add("trend[" + power + "]");
        }
        for (int index = 0; index < kExog; index++) {
            names.add("exog[" + index + "]");
        }
        for (int lag : arLags) {
            names.add("ar.L" + lag);
        }
        for (int lag : maLags) {
            names.add("ma.L" + lag);
        }
        for (int lag : seasonalArLags) {
            names.add("seasonalAr.L" + lag);
        }
        for (int lag : seasonalMaLags) {
            names.add("seasonalMa.L" + lag);
        }
        for (int index = 0; index < exogVarianceCount; index++) {
            names.add((concentrateScale ? "snr.x" : "var.x") + (index + 1));
        }
        if (measurementError) {
            names.add(concentrateScale ? "snr.measurement_error" : "var.measurement_error");
        }
        if (!concentrateScale) {
            names.add("sigma2");
        }
        return names.toArray(String[]::new);
    }

    private static CSSStart condSumSquare(int[] arLags,
                                          int[] maLags,
                                          double[] endogenous,
                                          double[][] trend) {
        int arOrder = arLags.length;
        int maOrder = maLags.length;
        int maxArLag = lastLag(arLags);
        int maxMaLag = lastLag(maLags);
        int kTrend = trend == null || trend.length == 0 ? 0 : trend[0].length;
        int k = 2 * maxMaLag;
        int r = Math.max(3 * maxMaLag, maxArLag);
        int paramCount = kTrend + arOrder + maOrder;
        if (paramCount == 0) {
            return new CSSStart(new double[0], new double[0], new double[0], Double.NaN);
        }

        try {
            double[] residuals = null;
            if (maOrder > 0) {
                if (endogenous.length <= k) {
                    throw new IllegalArgumentException("not enough observations for MA start");
                }
                double[] y = Arrays.copyOfRange(endogenous, k, endogenous.length);
                double[][] x = lagMatrixBoth(endogenous, k);
                residuals = fitOLS(y, x).residuals();
            }

            if (endogenous.length <= r) {
                throw new IllegalArgumentException("not enough observations for CSS start");
            }
            double[] y = Arrays.copyOfRange(endogenous, r, endogenous.length);
            double[][] design = new double[y.length][];
            for (int row = 0; row < y.length; row++) {
                design[row] = trend != null ? Arrays.copyOf(trend[row], kTrend) : new double[0];
            }
            if (arOrder > 0) {
                double[][] arLagMatrix = lagMatrixForward(endogenous, maxArLag);
                for (int row = r; row < arLagMatrix.length; row++) {
                    for (int lag = 0; lag < arOrder; lag++) {
                        design[row - r] = append(design[row - r], arLagMatrix[row][arLags[lag] - 1]);
                    }
                }
            }
            if (maOrder > 0) {
                double[][] maLagMatrix = lagMatrixForward(residuals, maxMaLag);
                for (int row = r - k; row < maLagMatrix.length; row++) {
                    for (int lag = 0; lag < maOrder; lag++) {
                        design[row - (r - k)] = append(design[row - (r - k)], maLagMatrix[row][maLags[lag] - 1]);
                    }
                }
            }
            OLSFit fit = fitOLS(y, design);
            int offset = 0;
            double[] trendParams = kTrend == 0 ? new double[0] : Arrays.copyOfRange(fit.params(), offset, offset + kTrend);
            offset += kTrend;
            double[] arParams = arOrder == 0 ? new double[0] : Arrays.copyOfRange(fit.params(), offset, offset + arOrder);
            offset += arOrder;
            double[] maParams = maOrder == 0 ? new double[0] : Arrays.copyOfRange(fit.params(), offset, offset + maOrder);
            return new CSSStart(arParams, maParams, trendParams, variance(fit.residuals()));
        } catch (RuntimeException ex) {
            double[] residuals = demeaned(endogenous);
            return new CSSStart(fillZeros(arOrder), fillZeros(maOrder), fillZeros(kTrend), variance(residuals));
        }
    }

    private static OLSFit fitOLS(double[] y, double[][] rows) {
        if (rows == null || rows.length == 0 || rows[0].length == 0) {
            return new OLSFit(new double[0], Arrays.copyOf(y, y.length));
        }
        int n = y.length;
        int k = rows[0].length;
        double[] x = new double[n * k];
        for (int row = 0; row < n; row++) {
            System.arraycopy(rows[row], 0, x, row * k, k);
        }
        OLS ols = new OLS(y, x, n, k, false, 0).fit();
        return new OLSFit(Arrays.copyOf(ols.params(), k), Arrays.copyOf(ols.residual(false), n));
    }

    private static double[][] lagMatrixBoth(double[] series, int maxLag) {
        if (maxLag <= 0) {
            return new double[series.length][0];
        }
        double[][] lags = new double[Math.max(0, series.length - maxLag)][maxLag];
        for (int row = maxLag; row < series.length; row++) {
            for (int lag = 1; lag <= maxLag; lag++) {
                lags[row - maxLag][lag - 1] = series[row - lag];
            }
        }
        return lags;
    }

    private static double[][] lagMatrixForward(double[] series, int maxLag) {
        if (maxLag <= 0) {
            return new double[series.length][0];
        }
        double[][] lags = new double[series.length][maxLag];
        for (int row = 0; row < series.length; row++) {
            for (int lag = 1; lag <= maxLag; lag++) {
                int index = row - lag;
                lags[row][lag - 1] = index >= 0 ? series[index] : 0.0;
            }
        }
        return lags;
    }

    private static boolean containsNaN(double[] values) {
        for (double value : values) {
            if (Double.isNaN(value)) {
                return true;
            }
        }
        return false;
    }

    private static double[][] copy2d(double[][] source) {
        if (source == null) {
            return null;
        }
        double[][] copy = new double[source.length][];
        for (int row = 0; row < source.length; row++) {
            copy[row] = source[row] == null ? null : Arrays.copyOf(source[row], source[row].length);
        }
        return copy;
    }

    private static double[] append(double[] source, double value) {
        double[] appended = Arrays.copyOf(source, source.length + 1);
        appended[source.length] = value;
        return appended;
    }

    private static double[] demeaned(double[] values) {
        double mean = 0.0;
        if (values.length > 0) {
            for (double value : values) {
                mean += value;
            }
            mean /= values.length;
        }
        double[] residuals = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            residuals[i] = values[i] - mean;
        }
        return residuals;
    }

    private static double variance(double[] values) {
        if (values.length == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.length;
        double sum = 0.0;
        for (double value : values) {
            double delta = value - mean;
            sum += delta * delta;
        }
        return sum / Math.max(1, values.length - 1);
    }

    private static double[] concat(double[]... arrays) {
        int length = 0;
        for (double[] array : arrays) {
            length += array.length;
        }
        double[] result = new double[length];
        int offset = 0;
        for (double[] array : arrays) {
            System.arraycopy(array, 0, result, offset, array.length);
            offset += array.length;
        }
        return result;
    }

    private static double[] fillZeros(int length) {
        return new double[Math.max(0, length)];
    }

    private static double[] fillOnes(int length) {
        double[] ones = new double[Math.max(0, length)];
        Arrays.fill(ones, 1.0);
        return ones;
    }

    private static PreparedData dropMissing(double[] endogenous, double[][] exogenous, double[][] trend) {
        if (!containsNaN(endogenous)) {
            return new PreparedData(endogenous, exogenous, trend);
        }
        double[] keptEndog = Arrays.copyOf(endogenous, endogenous.length);
        double[][] keptExog = copy2d(exogenous);
        double[][] keptTrend = copy2d(trend);
        int kept = 0;
        for (int row = 0; row < keptEndog.length; row++) {
            if (Double.isNaN(keptEndog[row])) {
                continue;
            }
            keptEndog[kept] = keptEndog[row];
            if (keptExog != null) {
                keptExog[kept] = keptExog[row];
            }
            if (keptTrend != null) {
                keptTrend[kept] = keptTrend[row];
            }
            kept++;
        }
        keptEndog = Arrays.copyOf(keptEndog, kept);
        if (keptExog != null) {
            keptExog = Arrays.copyOf(keptExog, kept);
        }
        if (keptTrend != null) {
            keptTrend = Arrays.copyOf(keptTrend, kept);
        }
        return new PreparedData(keptEndog, keptExog, keptTrend);
    }

    private static DifferencedData differenceData(double[] endogenous,
                                                  double[][] exogenous,
                                                  int diff,
                                                  int seasonalDiff,
                                                  int seasonalPeriod) {
        double[] differencedEndog = Arrays.copyOf(endogenous, endogenous.length);
        double[][] differencedExog = copy2d(exogenous);
        for (int i = 0; i < seasonalDiff; i++) {
            differencedEndog = difference(differencedEndog, seasonalPeriod);
            if (differencedExog != null) {
                differencedExog = difference(differencedExog, seasonalPeriod);
            }
        }
        for (int i = 0; i < diff; i++) {
            differencedEndog = difference(differencedEndog, 1);
            if (differencedExog != null) {
                differencedExog = difference(differencedExog, 1);
            }
        }
        return new DifferencedData(differencedEndog, differencedExog);
    }

    private static double[] difference(double[] values, int lag) {
        if (lag <= 0) {
            return Arrays.copyOf(values, values.length);
        }
        if (values.length <= lag) {
            return new double[0];
        }
        double[] differenced = new double[values.length - lag];
        for (int i = lag; i < values.length; i++) {
            differenced[i - lag] = values[i] - values[i - lag];
        }
        return differenced;
    }

    private static double[][] difference(double[][] values, int lag) {
        if (values == null) {
            return null;
        }
        if (lag <= 0) {
            return copy2d(values);
        }
        if (values.length <= lag) {
            return new double[0][];
        }
        double[][] differenced = new double[values.length - lag][];
        for (int row = lag; row < values.length; row++) {
            double[] current = values[row];
            double[] prior = values[row - lag];
            double[] delta = new double[current.length];
            for (int col = 0; col < current.length; col++) {
                delta[col] = current[col] - prior[col];
            }
            differenced[row - lag] = delta;
        }
        return differenced;
    }

    private static int lastLag(int[] lags) {
        return lags.length == 0 ? 0 : lags[lags.length - 1];
    }

    private static int stateDiff(Meta meta) {
        return meta.spec().simpleDifferencing() ? 0 : meta.spec().order().integration();
    }

    private static int stateSeasonalDiff(Meta meta) {
        return meta.spec().simpleDifferencing() ? 0 : meta.spec().seasonalOrder().integration();
    }

    private static int exogStateStart(Meta meta) {
        return meta.totalDiff() + meta.totalOrder();
    }

    private static double[] stabilizeInitialBlock(double[] params, boolean autoregressive) {
        if (params.length == 0) {
            return params;
        }
        double sign = autoregressive ? 1.0 : -1.0;
        double[] unconstrained = unconstrainStationary(sign, params);
        if (!allFinite(unconstrained)) {
            return fillZeros(params.length);
        }
        double[] reconstrained = constrainStationary(sign, unconstrained);
        return allFinite(reconstrained) ? reconstrained : fillZeros(params.length);
    }

    private static double[] toArray(ArrayList<Double> values) {
        double[] array = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }
        return array;
    }

    record Meta(SARIMAXSpec spec,
                double[] endog,
                double[][] exog,
                int[] trendPowers,
                double[][] trendData,
                int nobs,
                int kExog,
                int kTrend,
                int reducedArOrder,
                int reducedMaOrder,
                int totalDiff,
                int totalOrder,
                int kStates,
                int kPosdef,
                boolean stateRegression,
                boolean timeVaryingRegression,
                String[] parameterNames,
                double[] startParams,
                StateSpaceModel templateModel) {
    }

    record PreparedData(double[] endog, double[][] exog, double[][] trend) {
    }

    static final class StationaryTransformWorkspace {
        private double[] reflection = new double[0];
        private double[] matrix = new double[0];

        private void ensure(int order) {
            if (reflection.length < order) {
                reflection = new double[order];
            }
            int matrixLength = order * order;
            if (matrix.length < matrixLength) {
                matrix = new double[matrixLength];
            }
        }
    }

    record OLSFit(double[] params, double[] residuals) {
    }

    record CSSStart(double[] arParams, double[] maParams, double[] trendParams, double variance) {
    }

    record DifferencedData(double[] endog, double[][] exog) {
    }
}