package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class VARMAXSupport {

    private static final double MIN_VARIANCE = 1e-8;
    private static final double STABILITY_BOUND = 0.95;

    private VARMAXSupport() {
    }

    static Meta analyze(VARMAXSpec spec) {
        double[][] endog = spec.endog();
        double[][] exog = spec.exog();
        int nobs = endog.length;
        int kEndog = endog[0].length;
        int kExog = exog == null ? 0 : exog[0].length;
        int kTrend = spec.trendPowers().length;
        int kAr = spec.order().autoregressive();
        int kMa = spec.order().movingAverage();
        int minAr = Math.max(kAr, 1);
        int kOrder = minAr + kMa;
        int kStates = kEndog * kOrder;
        int kPosdef = kEndog;
        ParameterLayout layout = ParameterLayout.of(spec, kEndog, kExog, kTrend);
        String[] parameterNames = buildParameterNames(spec, layout, kEndog, kExog, kTrend, kAr, kMa);
        double[][] trendData = buildTrendData(spec.trendPowers(), nobs + 1, spec.trendOffset());

        Meta draft = new Meta(spec, endog, exog, trendData, nobs, kEndog, kExog, kTrend,
            kAr, kMa, minAr, kOrder, kStates, kPosdef, layout, parameterNames, null, null);
        KalmanSSM template = buildTemplateModel(draft);
        double[] startParams = initialParams(draft);
        return new Meta(spec, endog, exog, trendData, nobs, kEndog, kExog, kTrend,
            kAr, kMa, minAr, kOrder, kStates, kPosdef, layout, parameterNames, startParams, template);
    }

    static Meta extend(Meta base, VARMAXSpec spec) {
        double[][] endog = spec.endog();
        double[][] exog = spec.exog();
        int kExog = exog == null ? 0 : exog[0].length;
        if (endog[0].length != base.kEndog()) {
            throw new IllegalArgumentException("extended endog dimension must match the original model");
        }
        if (kExog != base.kExog()) {
            throw new IllegalArgumentException("extended exog dimension must match the original model");
        }
        double[][] trendData = buildTrendData(base.spec().trendPowers(), endog.length + 1, spec.trendOffset());
        Meta draft = new Meta(spec, endog, exog, trendData, endog.length, base.kEndog(), base.kExog(), base.kTrend(),
            base.kAr(), base.kMa(), base.minAr(), base.kOrder(), base.kStates(), base.kPosdef(),
            base.parameterLayout(), base.parameterNames(), base.startParams(), null);
        KalmanSSM template = buildTemplateModel(draft);
        return new Meta(spec, endog, exog, trendData, endog.length, base.kEndog(), base.kExog(), base.kTrend(),
            base.kAr(), base.kMa(), base.minAr(), base.kOrder(), base.kStates(), base.kPosdef(),
            base.parameterLayout(), base.parameterNames(), base.startParams(), template);
    }

    static KalmanSSM buildTemplateModel(Meta meta) {
        double[] endog = flattenEndog(meta.endog());
        boolean[] missing = new boolean[meta.nobs() * meta.kEndog()];
        int[] nmissing = new int[meta.nobs()];
        for (int t = 0; t < meta.nobs(); t++) {
            for (int i = 0; i < meta.kEndog(); i++) {
                if (Double.isNaN(meta.endog()[t][i])) {
                    missing[t * meta.kEndog() + i] = true;
                    nmissing[t]++;
                }
            }
        }
        return KalmanSSM.builder(meta.kEndog(), meta.kStates(), meta.kPosdef(), meta.nobs())
            .design(initDesign(meta), false)
            .obsIntercept(meta.spec().observationIntercept(), false)
            .obsCov(new double[meta.kEndog() * meta.kEndog()], false)
            .transition(initTransition(meta), false)
            .stateIntercept(new double[stateInterceptLength(meta)], stateInterceptTimeVarying(meta))
            .selection(initSelection(meta), false)
            .stateCovariance(new double[meta.kPosdef() * meta.kPosdef()], false)
            .endog(endog)
            .missing(missing, nmissing)
            .build();
    }

    static double[] initialParams(Meta meta) {
        ParameterLayout layout = meta.parameterLayout();
        double[] params = new double[layout.count()];
        double[][] prepared = fillMissing(meta.endog());
        double[][] adjusted = subtractObservationIntercept(prepared, meta.spec().observationIntercept());
        double[][] deterministic = deterministicDesign(meta);
        double[][] residual = copyMatrix(adjusted);
        double[][] trendCoef = new double[meta.kEndog()][meta.kTrend()];
        double[][] exogCoef = new double[meta.kEndog()][meta.kExog()];

        int deterministicWidth = deterministic.length == 0 ? 0 : deterministic[0].length;
        if (deterministicWidth > 0 && meta.nobs() > deterministicWidth) {
            double[] y = new double[meta.nobs()];
            double[] xBase = flattenRows(deterministic, 0, meta.nobs(), deterministicWidth);
            for (int equation = 0; equation < meta.kEndog(); equation++) {
                copyColumn(adjusted, equation, y);
                double[] beta = safeOls(y, xBase.clone(), meta.nobs(), deterministicWidth);
                for (int col = 0; col < deterministicWidth; col++) {
                    double value = beta[col];
                    if (col < meta.kTrend()) {
                        trendCoef[equation][col] = value;
                    } else {
                        exogCoef[equation][col - meta.kTrend()] = value;
                    }
                }
                for (int t = 0; t < meta.nobs(); t++) {
                    double fitted = 0.0;
                    for (int col = 0; col < deterministicWidth; col++) {
                        fitted += deterministic[t][col] * beta[col];
                    }
                    residual[t][equation] = adjusted[t][equation] - fitted;
                }
            }
        }

        double[][][] ar = estimateVAR(residual, meta.kAr());
        if (meta.kAr() > 0) {
            double[][] sum = sumMatrices(ar, meta.kEndog());
            double[][] transform = identityMinus(sum, meta.kEndog());
            trendCoef = multiply(transform, trendCoef, meta.kEndog(), meta.kTrend());
            exogCoef = multiply(transform, exogCoef, meta.kEndog(), meta.kExog());
            putCoefficientMatrices(params, layout.arOffset(), ar, meta.kEndog(), meta.kAr());
        }
        if (meta.kMa() > 0) {
            Arrays.fill(params, layout.maOffset(), layout.maOffset() + layout.maLength(), 0.0);
        }
        putEquationBlock(params, layout.trendOffset(), trendCoef, meta.kEndog(), meta.kTrend());
        putEquationBlock(params, layout.regressionOffset(), exogCoef, meta.kEndog(), meta.kExog());

        double[][] innovation = meta.kAr() > 0 ? varResiduals(residual, ar, meta.kAr()) : residual;
        double[] covariance = covariance(innovation, meta.kEndog());
        putStateCovarianceStart(params, layout, covariance, meta.kEndog(), meta.spec().errorCovariance());
        if (meta.spec().measurementError()) {
            for (int i = 0; i < meta.kEndog(); i++) {
                params[layout.obsCovOffset() + i] = Math.max(MIN_VARIANCE, covariance[i * meta.kEndog() + i] * 0.1);
            }
        }
        return params;
    }

    static void updateModel(Meta meta, double[] params, KalmanSSM model) {
        ParameterLayout layout = meta.parameterLayout();
        clearDynamicBlocks(meta, model);
        updateStateIntercept(meta, params, model);
        updateTransition(meta, params, model);
        updateStateCovariance(meta, params, model);
        updateObservationCovariance(meta, params, model);
    }

    static double[] transformParams(Meta meta, double[] unconstrained) {
        double[] constrained = Arrays.copyOf(unconstrained, unconstrained.length);
        ParameterLayout layout = meta.parameterLayout();
        if (meta.spec().hasOption(VARMAXOption.ENFORCE_STATIONARITY) && meta.kAr() > 0) {
            constrainCoefficientBlock(constrained,
                layout.arOffset(),
                meta.kEndog(),
                meta.kAr(),
                stateCovarianceForTransform(meta, unconstrained, layout));
        }
        if (meta.spec().hasOption(VARMAXOption.ENFORCE_INVERTIBILITY) && meta.kMa() > 0) {
            constrainCoefficientBlock(constrained,
                layout.maOffset(),
                meta.kEndog(),
                meta.kMa(),
                identity(meta.kEndog()));
        }
        if (meta.spec().errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < meta.kEndog(); i++) {
                double value = unconstrained[layout.stateCovOffset() + i];
                constrained[layout.stateCovOffset() + i] = value * value;
            }
        }
        if (meta.spec().measurementError()) {
            for (int i = 0; i < meta.kEndog(); i++) {
                double value = unconstrained[layout.obsCovOffset() + i];
                constrained[layout.obsCovOffset() + i] = value * value;
            }
        }
        return constrained;
    }

    static double[] untransformParams(Meta meta, double[] constrained) {
        double[] unconstrained = Arrays.copyOf(constrained, constrained.length);
        ParameterLayout layout = meta.parameterLayout();
        if (meta.spec().hasOption(VARMAXOption.ENFORCE_STATIONARITY) && meta.kAr() > 0) {
            unconstrainCoefficientBlock(unconstrained,
                layout.arOffset(),
                meta.kEndog(),
                meta.kAr(),
                stateCovarianceFromConstrained(meta, constrained, layout));
        }
        if (meta.spec().hasOption(VARMAXOption.ENFORCE_INVERTIBILITY) && meta.kMa() > 0) {
            unconstrainCoefficientBlock(unconstrained,
                layout.maOffset(),
                meta.kEndog(),
                meta.kMa(),
                identity(meta.kEndog()));
        }
        if (meta.spec().errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < meta.kEndog(); i++) {
                unconstrained[layout.stateCovOffset() + i] = Math.sqrt(Math.max(0.0, constrained[layout.stateCovOffset() + i]));
            }
        }
        if (meta.spec().measurementError()) {
            for (int i = 0; i < meta.kEndog(); i++) {
                unconstrained[layout.obsCovOffset() + i] = Math.sqrt(Math.max(0.0, constrained[layout.obsCovOffset() + i]));
            }
        }
        return unconstrained;
    }

    static double[][] buildTrendData(int[] trendPowers, int nobs, int trendOffset) {
        if (trendPowers.length == 0) {
            return new double[nobs][0];
        }
        double[][] trend = new double[nobs][trendPowers.length];
        for (int t = 0; t < nobs; t++) {
            double x = trendOffset + t;
            for (int col = 0; col < trendPowers.length; col++) {
                trend[t][col] = Math.pow(x, trendPowers[col]);
            }
        }
        return trend;
    }

    static double[][] appendMissingRows(double[][] base, int requiredLength) {
        if (requiredLength <= base.length) {
            return copyMatrix(base);
        }
        double[][] extended = new double[requiredLength][base[0].length];
        for (int t = 0; t < base.length; t++) {
            extended[t] = base[t].clone();
        }
        for (int t = base.length; t < requiredLength; t++) {
            Arrays.fill(extended[t], Double.NaN);
        }
        return extended;
    }

    static double[] flattenEndog(double[][] endog) {
        double[] flat = new double[endog.length * endog[0].length];
        for (int t = 0; t < endog.length; t++) {
            System.arraycopy(endog[t], 0, flat, t * endog[0].length, endog[0].length);
        }
        return flat;
    }

    private static int stateInterceptLength(Meta meta) {
        return stateInterceptTimeVarying(meta) ? meta.nobs() * meta.kStates() : meta.kStates();
    }

    static boolean stateInterceptTimeVarying(Meta meta) {
        return meta.kExog() > 0 || hasNonConstantTrend(meta);
    }

    private static boolean hasNonConstantTrend(Meta meta) {
        for (int power : meta.spec().trendPowers()) {
            if (power != 0) {
                return true;
            }
        }
        return false;
    }

    private static double[] initDesign(Meta meta) {
        double[] design = new double[meta.kEndog() * meta.kStates()];
        for (int i = 0; i < meta.kEndog(); i++) {
            design[i * meta.kStates() + i] = 1.0;
        }
        return design;
    }

    private static double[] initTransition(Meta meta) {
        double[] transition = new double[meta.kStates() * meta.kStates()];
        int k = meta.kEndog();
        for (int lag = 1; lag < meta.kAr(); lag++) {
            for (int i = 0; i < k; i++) {
                transition[(lag * k + i) * meta.kStates() + (lag - 1) * k + i] = 1.0;
            }
        }
        for (int lag = 1; lag < meta.kMa(); lag++) {
            int rowBase = (meta.minAr() + lag) * k;
            int colBase = (meta.minAr() + lag - 1) * k;
            for (int i = 0; i < k; i++) {
                transition[(rowBase + i) * meta.kStates() + colBase + i] = 1.0;
            }
        }
        return transition;
    }

    private static double[] initSelection(Meta meta) {
        double[] selection = new double[meta.kStates() * meta.kPosdef()];
        for (int i = 0; i < meta.kEndog(); i++) {
            selection[i * meta.kPosdef() + i] = 1.0;
        }
        if (meta.kMa() > 0) {
            int rowBase = meta.minAr() * meta.kEndog();
            for (int i = 0; i < meta.kEndog(); i++) {
                selection[(rowBase + i) * meta.kPosdef() + i] = 1.0;
            }
        }
        return selection;
    }

    private static void clearDynamicBlocks(Meta meta, KalmanSSM model) {
        Arrays.fill(model.stateInterceptData(), 0.0);
        double[] transition = model.transitionData();
        double[] base = initTransition(meta);
        System.arraycopy(base, 0, transition, model.transitionOffset(0), base.length);
        Arrays.fill(model.obsCovData(), 0.0);
        Arrays.fill(model.stateCovarianceData(), 0.0);
    }

    private static void updateStateIntercept(Meta meta, double[] params, KalmanSSM model) {
        ParameterLayout layout = meta.parameterLayout();
        boolean timeVarying = stateInterceptTimeVarying(meta);
        int periods = timeVarying ? meta.nobs() : 1;
        for (int t = 0; t < periods; t++) {
            int sourceTime = timeVarying ? t + 1 : 0;
            int offset = model.stateInterceptOffset(t);
            if (meta.kExog() > 0 && sourceTime >= meta.exog().length) {
                Arrays.fill(model.stateInterceptData(), offset, offset + meta.kEndog(), Double.NaN);
                continue;
            }
            for (int equation = 0; equation < meta.kEndog(); equation++) {
                double value = 0.0;
                for (int trend = 0; trend < meta.kTrend(); trend++) {
                    value += meta.trendData()[sourceTime][trend]
                        * params[layout.trendOffset() + equation * meta.kTrend() + trend];
                }
                if (meta.kExog() > 0 && meta.exog() != null) {
                    for (int exog = 0; exog < meta.kExog(); exog++) {
                        value += meta.exog()[sourceTime][exog]
                            * params[layout.regressionOffset() + equation * meta.kExog() + exog];
                    }
                }
                model.stateInterceptData()[offset + equation] = value;
            }
        }
    }

    private static void updateTransition(Meta meta, double[] params, KalmanSSM model) {
        ParameterLayout layout = meta.parameterLayout();
        double[] transition = model.transitionData();
        int offset = model.transitionOffset(0);
        int k = meta.kEndog();
        if (meta.kAr() > 0) {
            for (int equation = 0; equation < k; equation++) {
                for (int lag = 0; lag < meta.kAr(); lag++) {
                    for (int source = 0; source < k; source++) {
                        transition[offset + equation * meta.kStates() + lag * k + source] =
                            params[layout.arOffset() + equation * meta.kAr() * k + lag * k + source];
                    }
                }
            }
            for (int equation = 0; equation < k; equation++) {
                for (int lag = 0; lag < meta.kMa(); lag++) {
                    for (int source = 0; source < k; source++) {
                        transition[offset + equation * meta.kStates() + meta.kAr() * k + lag * k + source] =
                            params[layout.maOffset() + equation * meta.kMa() * k + lag * k + source];
                    }
                }
            }
        } else {
            for (int equation = 0; equation < k; equation++) {
                for (int lag = 0; lag < meta.kMa(); lag++) {
                    for (int source = 0; source < k; source++) {
                        transition[offset + equation * meta.kStates() + k + lag * k + source] =
                            params[layout.maOffset() + equation * meta.kMa() * k + lag * k + source];
                    }
                }
            }
        }
    }

    private static void updateStateCovariance(Meta meta, double[] params, KalmanSSM model) {
        ParameterLayout layout = meta.parameterLayout();
        double[] stateCov = model.stateCovarianceData();
        int offset = model.stateCovarianceOffset(0);
        int k = meta.kEndog();
        if (meta.spec().errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < k; i++) {
                stateCov[offset + i * k + i] = params[layout.stateCovOffset() + i];
            }
        } else {
            double[] lower = new double[k * k];
            int cursor = layout.stateCovOffset();
            for (int row = 0; row < k; row++) {
                for (int col = 0; col <= row; col++) {
                    lower[row * k + col] = params[cursor++];
                }
            }
            for (int row = 0; row < k; row++) {
                for (int col = 0; col < k; col++) {
                    double value = 0.0;
                    for (int inner = 0; inner <= Math.min(row, col); inner++) {
                        value += lower[row * k + inner] * lower[col * k + inner];
                    }
                    stateCov[offset + row * k + col] = value;
                }
            }
        }
    }

    private static void updateObservationCovariance(Meta meta, double[] params, KalmanSSM model) {
        if (!meta.spec().measurementError()) {
            return;
        }
        ParameterLayout layout = meta.parameterLayout();
        double[] obsCov = model.obsCovData();
        int offset = model.obsCovOffset(0);
        for (int i = 0; i < meta.kEndog(); i++) {
            obsCov[offset + i * meta.kEndog() + i] = params[layout.obsCovOffset() + i];
        }
    }

    private static double[][] deterministicDesign(Meta meta) {
        int width = meta.kTrend() + meta.kExog();
        if (width == 0) {
            return new double[0][0];
        }
        double[][] design = new double[meta.nobs()][width];
        for (int t = 0; t < meta.nobs(); t++) {
            if (meta.kTrend() > 0) {
                System.arraycopy(meta.trendData()[t], 0, design[t], 0, meta.kTrend());
            }
            if (meta.kExog() > 0) {
                System.arraycopy(meta.exog()[t], 0, design[t], meta.kTrend(), meta.kExog());
            }
        }
        return design;
    }

    private static double[][] fillMissing(double[][] source) {
        double[][] filled = copyMatrix(source);
        int nobs = filled.length;
        int k = filled[0].length;
        for (int col = 0; col < k; col++) {
            double mean = 0.0;
            int count = 0;
            for (int t = 0; t < nobs; t++) {
                if (Double.isFinite(filled[t][col])) {
                    mean += filled[t][col];
                    count++;
                }
            }
            mean = count == 0 ? 0.0 : mean / count;
            double last = mean;
            for (int t = 0; t < nobs; t++) {
                if (Double.isFinite(filled[t][col])) {
                    last = filled[t][col];
                } else {
                    filled[t][col] = last;
                }
            }
            double next = mean;
            for (int t = nobs - 1; t >= 0; t--) {
                if (Double.isFinite(source[t][col])) {
                    next = source[t][col];
                } else if (!Double.isFinite(source[Math.min(t + 1, nobs - 1)][col])) {
                    filled[t][col] = next;
                }
            }
        }
        return filled;
    }

    private static double[][] subtractObservationIntercept(double[][] source, double[] observationIntercept) {
        double[][] adjusted = copyMatrix(source);
        for (int row = 0; row < adjusted.length; row++) {
            for (int col = 0; col < adjusted[row].length; col++) {
                adjusted[row][col] -= observationIntercept[col];
            }
        }
        return adjusted;
    }

    private static double[][][] estimateVAR(double[][] endog, int order) {
        int nobs = endog.length;
        int k = endog[0].length;
        double[][][] coefficients = new double[Math.max(order, 0)][k][k];
        if (order <= 0 || nobs <= order + k * order) {
            return coefficients;
        }
        int rows = nobs - order;
        int cols = k * order;
        double[] x = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            int t = row + order;
            for (int lag = 0; lag < order; lag++) {
                System.arraycopy(endog[t - lag - 1], 0, x, row * cols + lag * k, k);
            }
        }
        for (int equation = 0; equation < k; equation++) {
            double[] y = new double[rows];
            for (int row = 0; row < rows; row++) {
                y[row] = endog[row + order][equation];
            }
            double[] beta = safeOls(y, x.clone(), rows, cols);
            for (int lag = 0; lag < order; lag++) {
                for (int source = 0; source < k; source++) {
                    coefficients[lag][equation][source] = beta[lag * k + source];
                }
            }
        }
        return coefficients;
    }

    private static double[][] varResiduals(double[][] endog, double[][][] ar, int order) {
        if (order <= 0) {
            return copyMatrix(endog);
        }
        int rows = endog.length - order;
        int k = endog[0].length;
        if (rows <= 0) {
            return new double[][]{new double[k]};
        }
        double[][] residuals = new double[rows][k];
        for (int row = 0; row < rows; row++) {
            int t = row + order;
            for (int equation = 0; equation < k; equation++) {
                double fitted = 0.0;
                for (int lag = 0; lag < order; lag++) {
                    for (int source = 0; source < k; source++) {
                        fitted += ar[lag][equation][source] * endog[t - lag - 1][source];
                    }
                }
                residuals[row][equation] = endog[t][equation] - fitted;
            }
        }
        return residuals;
    }

    private static double[] safeOls(double[] y, double[] x, int rows, int cols) {
        try {
            return Regressor.ols(y, x, rows, cols,
                    Regressor.Opts.PINV, Regressor.Opts.NO_CONST, Regressor.Opts.PARAMS).params();
        } catch (RuntimeException ex) {
            return new double[cols];
        }
    }

    private static double[] covariance(double[][] values, int k) {
        double[] covariance = new double[k * k];
        if (values.length == 0) {
            for (int i = 0; i < k; i++) {
                covariance[i * k + i] = 1.0;
            }
            return covariance;
        }
        for (double[] row : values) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    covariance[i * k + j] += row[i] * row[j];
                }
            }
        }
        double scale = 1.0 / Math.max(1, values.length);
        for (int i = 0; i < covariance.length; i++) {
            covariance[i] *= scale;
        }
        for (int i = 0; i < k; i++) {
            covariance[i * k + i] = Math.max(MIN_VARIANCE, covariance[i * k + i]);
        }
        return covariance;
    }

    private static void putStateCovarianceStart(double[] params,
                                                ParameterLayout layout,
                                                double[] covariance,
                                                int k,
                                                VARMAXErrorCovariance type) {
        if (type == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < k; i++) {
                params[layout.stateCovOffset() + i] = Math.max(MIN_VARIANCE, covariance[i * k + i]);
            }
            return;
        }
        double[] lower = choleskyLower(covariance, k);
        int cursor = layout.stateCovOffset();
        for (int row = 0; row < k; row++) {
            for (int col = 0; col <= row; col++) {
                params[cursor++] = lower[row * k + col];
            }
        }
    }

    private static double[] choleskyLower(double[] covariance, int k) {
        double[] lower = new double[k * k];
        double jitter = 0.0;
        for (int attempt = 0; attempt < 5; attempt++) {
            Arrays.fill(lower, 0.0);
            boolean ok = true;
            for (int row = 0; row < k && ok; row++) {
                for (int col = 0; col <= row; col++) {
                    double sum = covariance[row * k + col];
                    if (row == col) {
                        sum += jitter + MIN_VARIANCE;
                    }
                    for (int inner = 0; inner < col; inner++) {
                        sum -= lower[row * k + inner] * lower[col * k + inner];
                    }
                    if (row == col) {
                        if (!(sum > 0.0) || !Double.isFinite(sum)) {
                            ok = false;
                            break;
                        }
                        lower[row * k + col] = Math.sqrt(sum);
                    } else {
                        lower[row * k + col] = sum / lower[col * k + col];
                    }
                }
            }
            if (ok) {
                return lower;
            }
            jitter = jitter == 0.0 ? 1e-6 : jitter * 10.0;
        }
        Arrays.fill(lower, 0.0);
        for (int i = 0; i < k; i++) {
            lower[i * k + i] = Math.sqrt(Math.max(MIN_VARIANCE, covariance[i * k + i]));
        }
        return lower;
    }

    private static void constrainCoefficientBlock(double[] params, int offset, int k, int order, double[] errorVariance) {
        if (order <= 0) {
            return;
        }
        try {
            double[][] coefficients = constrainStationaryMultivariate(params, offset, k, order, errorVariance);
            for (int lag = 0; lag < order; lag++) {
                double[] matrix = coefficients[lag];
                for (int equation = 0; equation < k; equation++) {
                    for (int source = 0; source < k; source++) {
                        params[offset + equation * order * k + lag * k + source] = matrix[equation * k + source];
                    }
                }
            }
        } catch (ArithmeticException ex) {
            stabilizeCoefficientBlock(params, offset, k, order);
        }
    }

    private static double[][] constrainStationaryMultivariate(double[] params,
                                                              int offset,
                                                              int k,
                                                              int order,
                                                              double[] errorVariance) {
        return constrainStationaryMultivariate(coefficientMatrices(params, offset, k, order), errorVariance, order, k);
    }

    private static double[][] constrainStationaryMultivariate(double[][] unconstrained,
                                                              double[] errorVariance,
                                                              int order,
                                                              int k) {
        double[][] partial = new double[order][];
        double[] eye = identity(k);
        for (int i = 0; i < order; i++) {
            double[] aaT = multiply(unconstrained[i], transpose(unconstrained[i], k, k), k, k, k);
            double[] factor = choleskyLower(add(eye, aaT));
            partial[i] = solveLower(factor, unconstrained[i], k, k);
        }
        return coefficientsFromMultivariatePacf(partial, regularizeVariance(errorVariance, k), order, k);
    }

    private static void unconstrainCoefficientBlock(double[] params, int offset, int k, int order, double[] errorVariance) {
        if (order <= 0) {
            return;
        }
        try {
            double[][] coefficients = unconstrainStationaryMultivariate(params, offset, k, order, errorVariance);
            for (int lag = 0; lag < order; lag++) {
                double[] matrix = coefficients[lag];
                for (int equation = 0; equation < k; equation++) {
                    for (int source = 0; source < k; source++) {
                        params[offset + equation * order * k + lag * k + source] = matrix[equation * k + source];
                    }
                }
            }
        } catch (ArithmeticException ex) {
            stabilizeCoefficientBlock(params, offset, k, order);
        }
    }

    private static double[][] unconstrainStationaryMultivariate(double[] params,
                                                                int offset,
                                                                int k,
                                                                int order,
                                                                double[] errorVariance) {
        double[][] constrained = coefficientMatrices(params, offset, k, order);
        double[][] partial = pacfFromCoefficients(constrained, regularizeVariance(errorVariance, k), order, k);
        double[][] unconstrained = new double[order][];
        double[] eye = identity(k);
        for (int i = 0; i < order; i++) {
            double[] ppT = multiply(partial[i], transpose(partial[i], k, k), k, k, k);
            double[] factor = choleskyLower(subtract(eye, ppT));
            unconstrained[i] = solveLower(factor, partial[i], k, k);
        }
        return refineUnconstrainedStationaryMultivariate(constrained, unconstrained, errorVariance, order, k);
    }

    private static double[][] refineUnconstrainedStationaryMultivariate(double[][] target,
                                                                        double[][] initial,
                                                                        double[] errorVariance,
                                                                        int order,
                                                                        int k) {
        int dimension = order * k * k;
        if (dimension == 0 || dimension > 64) {
            return initial;
        }
        double[] targetFlat = flattenCoefficientMatrices(target, order, k);
        double[] values = flattenCoefficientMatrices(initial, order, k);
        double[] residual = transformResidual(values, targetFlat, errorVariance, order, k);
        double best = maxAbs(residual);
        if (best <= 1e-10) {
            return unflattenCoefficientMatrices(values, order, k);
        }
        for (int iteration = 0; iteration < 12; iteration++) {
            double[] jacobian = new double[dimension * dimension];
            for (int col = 0; col < dimension; col++) {
                double original = values[col];
                double step = 1e-6 * Math.max(1.0, Math.abs(original));
                values[col] = original + step;
                double[] shifted = transformResidual(values, targetFlat, errorVariance, order, k);
                values[col] = original;
                for (int row = 0; row < dimension; row++) {
                    jacobian[row * dimension + col] = (shifted[row] - residual[row]) / step;
                }
            }
            double[] rhs = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                rhs[i] = -residual[i];
            }
            double[] step;
            try {
                step = solveLinear(jacobian, rhs, dimension);
            } catch (ArithmeticException ex) {
                return initial;
            }
            boolean accepted = false;
            double damping = 1.0;
            while (damping >= 1.0 / 64.0) {
                double[] candidate = values.clone();
                for (int i = 0; i < dimension; i++) {
                    candidate[i] += damping * step[i];
                }
                double[] candidateResidual = transformResidual(candidate, targetFlat, errorVariance, order, k);
                double candidateNorm = maxAbs(candidateResidual);
                if (candidateNorm < best) {
                    values = candidate;
                    residual = candidateResidual;
                    best = candidateNorm;
                    accepted = true;
                    break;
                }
                damping *= 0.5;
            }
            if (best <= 1e-10) {
                break;
            }
            if (!accepted) {
                return initial;
            }
        }
        return best <= 1e-7 ? unflattenCoefficientMatrices(values, order, k) : initial;
    }

    private static double[] transformResidual(double[] unconstrained,
                                              double[] target,
                                              double[] errorVariance,
                                              int order,
                                              int k) {
        double[][] matrices = unflattenCoefficientMatrices(unconstrained, order, k);
        double[] transformed = flattenCoefficientMatrices(
            constrainStationaryMultivariate(matrices, errorVariance, order, k), order, k);
        double[] residual = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            residual[i] = transformed[i] - target[i];
        }
        return residual;
    }

    private static double[] flattenCoefficientMatrices(double[][] matrices, int order, int k) {
        double[] values = new double[order * k * k];
        for (int lag = 0; lag < order; lag++) {
            System.arraycopy(matrices[lag], 0, values, lag * k * k, k * k);
        }
        return values;
    }

    private static double[][] unflattenCoefficientMatrices(double[] values, int order, int k) {
        double[][] matrices = new double[order][k * k];
        for (int lag = 0; lag < order; lag++) {
            System.arraycopy(values, lag * k * k, matrices[lag], 0, k * k);
        }
        return matrices;
    }

    private static double maxAbs(double[] values) {
        double max = 0.0;
        for (double value : values) {
            max = Math.max(max, Math.abs(value));
        }
        return max;
    }

    private static double[][] coefficientsFromMultivariatePacf(double[][] partial,
                                                               double[] initialVariance,
                                                               int order,
                                                               int k) {
        double scale = Math.pow(order + k, 10.0);
        double[] workingVariance = identity(k);
        for (int i = 0; i < k; i++) {
            workingVariance[i * k + i] = scale;
        }
        double[][] forwardVariances = new double[order + 1][];
        double[][] backwardVariances = new double[order + 1][];
        double[][] autocovariances = new double[order + 1][];
        double[][] forwardFactors = new double[order + 1][];
        double[][] backwardFactors = new double[order + 1][];
        forwardVariances[0] = workingVariance.clone();
        backwardVariances[0] = workingVariance.clone();
        autocovariances[0] = workingVariance.clone();
        forwardFactors[0] = choleskyLower(forwardVariances[0]);
        backwardFactors[0] = choleskyLower(backwardVariances[0]);

        double[][] forwards = new double[0][];
        double[][] backwards = new double[0][];
        for (int s = 0; s < order; s++) {
            double[][] prevForwards = forwards;
            double[][] prevBackwards = backwards;
            forwards = new double[s + 1][];
            backwards = new double[s + 1][];

            double[] forwardLast = multiply(forwardFactors[s],
                transpose(solveLowerTranspose(backwardFactors[s], transpose(partial[s], k, k), k, k), k, k),
                k, k, k);
            double[] backwardLast = multiply(backwardFactors[s],
                transpose(solveLowerTranspose(forwardFactors[s], partial[s], k, k), k, k),
                k, k, k);
            forwards[s] = forwardLast;
            backwards[s] = backwardLast;

            double[] tmp = multiply(forwardLast, backwardVariances[s], k, k, k);
            autocovariances[s + 1] = transpose(tmp, k, k);

            for (int j = 0; j < s; j++) {
                forwards[j] = subtract(prevForwards[j], multiply(forwardLast, prevBackwards[s - (j + 1)], k, k, k));
                backwards[j] = subtract(prevBackwards[j], multiply(backwardLast, prevForwards[s - (j + 1)], k, k, k));
                autocovariances[s + 1] = add(autocovariances[s + 1],
                    multiply(autocovariances[j + 1], transpose(prevForwards[s - (j + 1)], k, k), k, k, k));
            }

            forwardVariances[s + 1] = subtract(forwardVariances[s], multiply(tmp, transpose(forwards[s], k, k), k, k, k));
            backwardVariances[s + 1] = subtract(backwardVariances[s],
                multiply(multiply(backwards[s], forwardVariances[s], k, k, k), transpose(backwards[s], k, k), k, k, k));
            forwardFactors[s + 1] = choleskyLower(regularizeVariance(forwardVariances[s + 1], k));
            backwardFactors[s + 1] = choleskyLower(regularizeVariance(backwardVariances[s + 1], k));
        }

        double[] initialFactor = choleskyLower(regularizeVariance(initialVariance, k));
        double[] transformedFactor = choleskyLower(regularizeVariance(forwardVariances[order], k));
        double[] transform = multiply(initialFactor, inverseLower(transformedFactor, k), k, k, k);
        double[] inverseTransform = inverse(transform, k);
        for (int i = 0; i < order; i++) {
            forwards[i] = multiply(multiply(transform, forwards[i], k, k, k), inverseTransform, k, k, k);
        }
        return forwards;
    }

    private static double[][] pacfFromCoefficients(double[][] coefficients, double[] errorVariance, int order, int k) {
        double[][] autocovariances = autocovariancesFromCoefficients(coefficients, errorVariance, order, k);
        for (int i = 0; i < autocovariances.length; i++) {
            autocovariances[i] = transpose(autocovariances[i], k, k);
        }
        return pacfFromAutocovariances(autocovariances, order, k);
    }

    private static double[][] pacfFromAutocovariances(double[][] autocovariances, int order, int k) {
        double[][] forwards = new double[0][];
        double[][] backwards = new double[0][];
        double[][] partial = new double[order][];
        for (int s = 0; s < order; s++) {
            double[][] prevForwards = forwards;
            double[][] prevBackwards = backwards;
            forwards = new double[s + 1][];
            backwards = new double[s + 1][];

            double[] forwardVariance = autocovariances[0].clone();
            double[] backwardVariance = transpose(autocovariances[0], k, k);
            for (int j = 0; j < s; j++) {
                forwardVariance = subtract(forwardVariance, multiply(prevForwards[j], autocovariances[j + 1], k, k, k));
                backwardVariance = subtract(backwardVariance,
                    multiply(prevBackwards[j], transpose(autocovariances[j + 1], k, k), k, k, k));
            }
            double[] forwardFactor = choleskyLower(regularizeVariance(forwardVariance, k));
            double[] backwardFactor = choleskyLower(regularizeVariance(backwardVariance, k));

            if (s == 0) {
                forwards[0] = transpose(solveCholeskySystem(forwardFactor, autocovariances[1], k, k), k, k);
                backwards[0] = transpose(solveCholeskySystem(backwardFactor, transpose(autocovariances[1], k, k), k, k), k, k);
            } else {
                double[] tmp = transpose(autocovariances[s + 1], k, k);
                for (int j = 0; j < s; j++) {
                    tmp = subtract(tmp,
                        multiply(prevForwards[j], transpose(autocovariances[s - j], k, k), k, k, k));
                }
                forwards[s] = transpose(solveCholeskySystem(backwardFactor, transpose(tmp, k, k), k, k), k, k);
                backwards[s] = transpose(solveCholeskySystem(forwardFactor, tmp, k, k), k, k);
            }

            for (int j = 0; j < s; j++) {
                forwards[j] = subtract(prevForwards[j], multiply(forwards[s], prevBackwards[s - (j + 1)], k, k, k));
                backwards[j] = subtract(prevBackwards[j], multiply(backwards[s], prevForwards[s - (j + 1)], k, k, k));
            }
            partial[s] = solveLower(forwardFactor, multiply(forwards[s], backwardFactor, k, k, k), k, k);
        }
        return partial;
    }

    private static double[][] autocovariancesFromCoefficients(double[][] coefficients,
                                                              double[] errorVariance,
                                                              int order,
                                                              int k) {
        int dimension = order * k;
        double[] companion = new double[dimension * dimension];
        for (int lag = 0; lag < order; lag++) {
            double[] coefficient = coefficients[lag];
            for (int row = 0; row < k; row++) {
                System.arraycopy(coefficient, row * k, companion, row * dimension + lag * k, k);
            }
        }
        for (int lag = 1; lag < order; lag++) {
            for (int i = 0; i < k; i++) {
                companion[lag * k * dimension + (lag - 1) * k + i] = 1.0;
            }
        }
        double[] selectedVariance = new double[dimension * dimension];
        for (int row = 0; row < k; row++) {
            System.arraycopy(errorVariance, row * k, selectedVariance, row * dimension, k);
        }
        double[] stackedCovariance = solveDiscreteLyapunov(companion, selectedVariance, dimension);
        double[][] autocovariances = new double[order + 1][];
        int count = 0;
        for (; count < Math.min(order, order + 1); count++) {
            autocovariances[count] = block(stackedCovariance, dimension, 0, count * k, k, k);
        }
        double[] cross = stackedCovariance;
        while (count <= order) {
            cross = multiply(companion, cross, dimension, dimension, dimension);
            autocovariances[count++] = block(cross, dimension, 0, dimension - k, k, k);
        }
        return autocovariances;
    }

    private static double[][] coefficientMatrices(double[] params, int offset, int k, int order) {
        double[][] matrices = new double[order][k * k];
        for (int lag = 0; lag < order; lag++) {
            for (int equation = 0; equation < k; equation++) {
                for (int source = 0; source < k; source++) {
                    matrices[lag][equation * k + source] = params[offset + equation * order * k + lag * k + source];
                }
            }
        }
        return matrices;
    }

    private static double[] stateCovarianceForTransform(Meta meta, double[] params, ParameterLayout layout) {
        int k = meta.kEndog();
        double[] covariance = new double[k * k];
        if (meta.spec().errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < k; i++) {
                double value = params[layout.stateCovOffset() + i];
                covariance[i * k + i] = value * value;
            }
        } else {
            double[] lower = new double[k * k];
            int cursor = layout.stateCovOffset();
            for (int row = 0; row < k; row++) {
                for (int col = 0; col <= row; col++) {
                    lower[row * k + col] = params[cursor++];
                }
            }
            covariance = multiply(lower, transpose(lower, k, k), k, k, k);
        }
        return regularizeVariance(covariance, k);
    }

    private static double[] stateCovarianceFromConstrained(Meta meta, double[] params, ParameterLayout layout) {
        int k = meta.kEndog();
        double[] covariance = new double[k * k];
        if (meta.spec().errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (int i = 0; i < k; i++) {
                covariance[i * k + i] = Math.max(params[layout.stateCovOffset() + i], MIN_VARIANCE);
            }
        } else {
            double[] lower = new double[k * k];
            int cursor = layout.stateCovOffset();
            for (int row = 0; row < k; row++) {
                for (int col = 0; col <= row; col++) {
                    lower[row * k + col] = params[cursor++];
                }
            }
            covariance = multiply(lower, transpose(lower, k, k), k, k, k);
        }
        return regularizeVariance(covariance, k);
    }

    private static double[] regularizeVariance(double[] matrix, int k) {
        double[] copy = matrix.clone();
        for (int i = 0; i < k; i++) {
            copy[i * k + i] = Math.max(copy[i * k + i], MIN_VARIANCE);
        }
        return copy;
    }

    private static double[] identity(int n) {
        double[] eye = new double[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = 1.0;
        }
        return eye;
    }

    private static double[] add(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] + right[i];
        }
        return out;
    }

    private static double[] subtract(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] - right[i];
        }
        return out;
    }

    private static double[] multiply(double[] left, double[] right, int rows, int inner, int cols) {
        double[] out = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double sum = 0.0;
                for (int i = 0; i < inner; i++) {
                    sum += left[row * inner + i] * right[i * cols + col];
                }
                out[row * cols + col] = sum;
            }
        }
        return out;
    }

    private static double[] transpose(double[] matrix, int rows, int cols) {
        double[] out = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                out[col * rows + row] = matrix[row * cols + col];
            }
        }
        return out;
    }

    private static double[] choleskyLower(double[] matrix) {
        int n = (int) Math.round(Math.sqrt(matrix.length));
        double[] lower = new double[matrix.length];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col <= row; col++) {
                double sum = matrix[row * n + col];
                for (int i = 0; i < col; i++) {
                    sum -= lower[row * n + i] * lower[col * n + i];
                }
                if (row == col) {
                    if (!(sum > 0.0) || !Double.isFinite(sum)) {
                        throw new ArithmeticException("matrix is not positive definite");
                    }
                    lower[row * n + col] = Math.sqrt(sum);
                } else {
                    lower[row * n + col] = sum / lower[col * n + col];
                }
            }
        }
        return lower;
    }

    private static double[] solveLower(double[] lower, double[] rhs, int n, int columns) {
        double[] out = new double[n * columns];
        for (int col = 0; col < columns; col++) {
            for (int row = 0; row < n; row++) {
                double sum = rhs[row * columns + col];
                for (int i = 0; i < row; i++) {
                    sum -= lower[row * n + i] * out[i * columns + col];
                }
                out[row * columns + col] = sum / lower[row * n + row];
            }
        }
        return out;
    }

    private static double[] solveLowerTranspose(double[] lower, double[] rhs, int n, int columns) {
        double[] out = new double[n * columns];
        for (int col = 0; col < columns; col++) {
            for (int row = n - 1; row >= 0; row--) {
                double sum = rhs[row * columns + col];
                for (int i = row + 1; i < n; i++) {
                    sum -= lower[i * n + row] * out[i * columns + col];
                }
                out[row * columns + col] = sum / lower[row * n + row];
            }
        }
        return out;
    }

    private static double[] solveCholeskySystem(double[] lower, double[] rhs, int n, int columns) {
        return solveLowerTranspose(lower, solveLower(lower, rhs, n, columns), n, columns);
    }

    private static double[] solveDiscreteLyapunov(double[] transition, double[] variance, int n) {
        int dimension = n * n;
        double[] system = new double[dimension * dimension];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                int equation = row * n + col;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        int variable = i * n + j;
                        system[equation * dimension + variable] = (row == i && col == j ? 1.0 : 0.0)
                            - transition[row * n + i] * transition[col * n + j];
                    }
                }
            }
        }
        return solveLinear(system, variance.clone(), dimension);
    }

    private static double[] block(double[] matrix, int columns, int rowOffset, int colOffset, int rows, int cols) {
        double[] out = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(matrix, (rowOffset + row) * columns + colOffset, out, row * cols, cols);
        }
        return out;
    }

    private static double[] solveLinear(double[] matrix, double[] rhs, int n) {
        double[] a = matrix.clone();
        double[] b = rhs.clone();
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double pivotAbs = Math.abs(a[col * n + col]);
            for (int row = col + 1; row < n; row++) {
                double abs = Math.abs(a[row * n + col]);
                if (abs > pivotAbs) {
                    pivot = row;
                    pivotAbs = abs;
                }
            }
            if (!(pivotAbs > 0.0) || !Double.isFinite(pivotAbs)) {
                throw new ArithmeticException("linear system is singular");
            }
            if (pivot != col) {
                swapRows(a, n, pivot, col);
                double tmp = b[pivot];
                b[pivot] = b[col];
                b[col] = tmp;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = a[row * n + col] / a[col * n + col];
                a[row * n + col] = 0.0;
                for (int j = col + 1; j < n; j++) {
                    a[row * n + j] -= factor * a[col * n + j];
                }
                b[row] -= factor * b[col];
            }
        }
        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double sum = b[row];
            for (int col = row + 1; col < n; col++) {
                sum -= a[row * n + col] * x[col];
            }
            x[row] = sum / a[row * n + row];
        }
        return x;
    }

    private static double[] inverseLower(double[] lower, int n) {
        return solveLower(lower, identity(n), n, n);
    }

    private static double[] inverse(double[] matrix, int n) {
        double[] a = matrix.clone();
        double[] inverse = identity(n);
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double pivotAbs = Math.abs(a[col * n + col]);
            for (int row = col + 1; row < n; row++) {
                double abs = Math.abs(a[row * n + col]);
                if (abs > pivotAbs) {
                    pivot = row;
                    pivotAbs = abs;
                }
            }
            if (!(pivotAbs > 0.0) || !Double.isFinite(pivotAbs)) {
                throw new ArithmeticException("matrix is singular");
            }
            if (pivot != col) {
                swapRows(a, n, pivot, col);
                swapRows(inverse, n, pivot, col);
            }
            double scale = a[col * n + col];
            for (int j = 0; j < n; j++) {
                a[col * n + j] /= scale;
                inverse[col * n + j] /= scale;
            }
            for (int row = 0; row < n; row++) {
                if (row == col) {
                    continue;
                }
                double factor = a[row * n + col];
                for (int j = 0; j < n; j++) {
                    a[row * n + j] -= factor * a[col * n + j];
                    inverse[row * n + j] -= factor * inverse[col * n + j];
                }
            }
        }
        return inverse;
    }

    private static void swapRows(double[] matrix, int columns, int rowA, int rowB) {
        for (int col = 0; col < columns; col++) {
            double tmp = matrix[rowA * columns + col];
            matrix[rowA * columns + col] = matrix[rowB * columns + col];
            matrix[rowB * columns + col] = tmp;
        }
    }

    private static void stabilizeCoefficientBlock(double[] params, int offset, int k, int order) {
        if (order <= 0) {
            return;
        }
        double maxRow = 0.0;
        for (int equation = 0; equation < k; equation++) {
            double rowSum = 0.0;
            for (int lag = 0; lag < order; lag++) {
                for (int source = 0; source < k; source++) {
                    rowSum += Math.abs(params[offset + equation * order * k + lag * k + source]);
                }
            }
            maxRow = Math.max(maxRow, rowSum);
        }
        if (maxRow > STABILITY_BOUND) {
            double scale = STABILITY_BOUND / maxRow;
            for (int i = 0; i < k * k * order; i++) {
                params[offset + i] *= scale;
            }
        }
    }

    private static String[] buildParameterNames(VARMAXSpec spec,
                                                ParameterLayout layout,
                                                int kEndog,
                                                int kExog,
                                                int kTrend,
                                                int kAr,
                                                int kMa) {
        List<String> names = new ArrayList<>(layout.count());
        String[] endogNames = spec.endogNames();
        String[] exogNames = spec.exogNames();
        int[] trendPowers = spec.trendPowers();
        for (String endogName : endogNames) {
            for (int power : trendPowers) {
                if (power == 0) {
                    names.add("intercept." + endogName);
                } else if (power == 1) {
                    names.add("drift." + endogName);
                } else {
                    names.add("trend." + power + "." + endogName);
                }
            }
        }
        for (String equation : endogNames) {
            for (int lag = 1; lag <= kAr; lag++) {
                for (String source : endogNames) {
                    names.add("L" + lag + "." + source + "." + equation);
                }
            }
        }
        for (String equation : endogNames) {
            for (int lag = 1; lag <= kMa; lag++) {
                for (String source : endogNames) {
                    names.add("L" + lag + ".e(" + source + ")." + equation);
                }
            }
        }
        for (String equation : endogNames) {
            for (int exog = 0; exog < kExog; exog++) {
                names.add("beta." + exogNames[exog] + "." + equation);
            }
        }
        if (spec.errorCovariance() == VARMAXErrorCovariance.DIAGONAL) {
            for (String endogName : endogNames) {
                names.add("sigma2." + endogName);
            }
        } else {
            for (int row = 0; row < kEndog; row++) {
                for (int col = 0; col <= row; col++) {
                    names.add(row == col
                        ? "sqrt.var." + endogNames[row]
                        : "sqrt.cov." + endogNames[col] + "." + endogNames[row]);
                }
            }
        }
        if (spec.measurementError()) {
            for (String endogName : endogNames) {
                names.add("measurement_variance." + endogName);
            }
        }
        return names.toArray(String[]::new);
    }

    private static double[] column(double[][] matrix, int col) {
        double[] values = new double[matrix.length];
        copyColumn(matrix, col, values);
        return values;
    }

    private static void copyColumn(double[][] matrix, int col, double[] values) {
        for (int row = 0; row < matrix.length; row++) {
            values[row] = matrix[row][col];
        }
    }

    private static double[] flattenRows(double[][] matrix, int start, int rows, int cols) {
        double[] flat = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(matrix[start + row], 0, flat, row * cols, cols);
        }
        return flat;
    }

    private static double[][] copyMatrix(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }

    private static double[][] sumMatrices(double[][][] matrices, int k) {
        double[][] sum = new double[k][k];
        for (double[][] matrix : matrices) {
            for (int row = 0; row < k; row++) {
                for (int col = 0; col < k; col++) {
                    sum[row][col] += matrix[row][col];
                }
            }
        }
        return sum;
    }

    private static double[][] identityMinus(double[][] matrix, int k) {
        double[][] result = new double[k][k];
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                result[row][col] = (row == col ? 1.0 : 0.0) - matrix[row][col];
            }
        }
        return result;
    }

    private static double[][] multiply(double[][] left, double[][] right, int rows, int cols) {
        if (cols == 0) {
            return new double[rows][0];
        }
        double[][] result = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double value = 0.0;
                for (int inner = 0; inner < rows; inner++) {
                    value += left[row][inner] * right[inner][col];
                }
                result[row][col] = value;
            }
        }
        return result;
    }

    private static void putCoefficientMatrices(double[] params, int offset, double[][][] matrices, int k, int order) {
        for (int equation = 0; equation < k; equation++) {
            for (int lag = 0; lag < order; lag++) {
                for (int source = 0; source < k; source++) {
                    params[offset + equation * order * k + lag * k + source] = matrices[lag][equation][source];
                }
            }
        }
    }

    private static void putEquationBlock(double[] params, int offset, double[][] values, int equations, int width) {
        for (int equation = 0; equation < equations; equation++) {
            for (int col = 0; col < width; col++) {
                params[offset + equation * width + col] = values[equation][col];
            }
        }
    }

    record Meta(VARMAXSpec spec,
                double[][] endog,
                double[][] exog,
                double[][] trendData,
                int nobs,
                int kEndog,
                int kExog,
                int kTrend,
                int kAr,
                int kMa,
                int minAr,
                int kOrder,
                int kStates,
                int kPosdef,
                ParameterLayout parameterLayout,
                String[] parameterNames,
                double[] startParams,
                KalmanSSM templateModel) {
    }

    record ParameterLayout(int trendOffset,
                           int trendLength,
                           int arOffset,
                           int arLength,
                           int maOffset,
                           int maLength,
                           int regressionOffset,
                           int regressionLength,
                           int stateCovOffset,
                           int stateCovLength,
                           int obsCovOffset,
                           int obsCovLength,
                           int count) {

        static ParameterLayout of(VARMAXSpec spec, int kEndog, int kExog, int kTrend) {
            int offset = 0;
            int trendLength = kEndog * kTrend;
            int trendOffset = offset;
            offset += trendLength;
            int arLength = kEndog * kEndog * spec.order().autoregressive();
            int arOffset = offset;
            offset += arLength;
            int maLength = kEndog * kEndog * spec.order().movingAverage();
            int maOffset = offset;
            offset += maLength;
            int regressionLength = kEndog * kExog;
            int regressionOffset = offset;
            offset += regressionLength;
            int stateCovLength = spec.errorCovariance() == VARMAXErrorCovariance.DIAGONAL
                ? kEndog
                : kEndog * (kEndog + 1) / 2;
            int stateCovOffset = offset;
            offset += stateCovLength;
            int obsCovLength = spec.measurementError() ? kEndog : 0;
            int obsCovOffset = offset;
            offset += obsCovLength;
            return new ParameterLayout(trendOffset, trendLength, arOffset, arLength,
                maOffset, maLength, regressionOffset, regressionLength,
                stateCovOffset, stateCovLength, obsCovOffset, obsCovLength, offset);
        }
    }
}
