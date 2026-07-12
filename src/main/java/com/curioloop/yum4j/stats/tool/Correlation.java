package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.DoubleI;
import com.curioloop.yum4j.math.VectorOps;
import com.curioloop.yum4j.stats.BetaDistribution;
import com.curioloop.yum4j.stats.NormalDistribution;
import com.curioloop.yum4j.stats.StudentsTDistribution;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Pearson, Spearman, and Kendall correlation coefficients with p-values.
 *
 * <p>The scalar methods mirror the core SciPy {@code scipy.stats} behavior for
 * one-dimensional numeric inputs. Missing values are rejected by default; use
 * {@link MissingPolicy#DROP} to run a pairwise-complete calculation.</p>
 */
public final class Correlation {

    private static final NormalDistribution STANDARD_NORMAL = new NormalDistribution();
    private static final Comparator<DoubleI> RANK_COMPARATOR =
            (left, right) -> Double.compare(left.value(), right.value());

    private Correlation() {
    }

    /** Direction of the alternative hypothesis. */
    public enum Alternative {
        TWO_SIDED,
        LESS,
        GREATER
    }

    /** Kendall tau normalization variant. */
    public enum KendallVariant {
        B,
        C
    }

    /** Kendall p-value method. */
    public enum KendallMethod {
        AUTO,
        ASYMPTOTIC,
        EXACT
    }

    /** Scalar correlation result. */
    public record Result(double statistic, double pValue) {
    }

    /** Row-major pairwise correlation and p-value matrices. */
    public record MatrixResult(double[] statistic, double[] pValue, int variables) {
        public MatrixResult {
            if (variables <= 0) throw new IllegalArgumentException("variables must be positive");
            if (statistic.length != variables * variables) {
                throw new IllegalArgumentException("statistic length must equal variables * variables");
            }
            if (pValue.length != variables * variables) {
                throw new IllegalArgumentException("pValue length must equal variables * variables");
            }
        }

        public double statistic(int row, int column) {
            return statistic[row * variables + column];
        }

        public double pValue(int row, int column) {
            return pValue[row * variables + column];
        }
    }

    public static Result pearson(double[] left, double[] right) {
        return pearson(left, right, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static Result pearson(double[] left, double[] right, Alternative alternative) {
        return pearson(left, right, alternative, MissingPolicy.RAISE);
    }

    public static Result pearson(double[] left, double[] right, Alternative alternative, MissingPolicy missing) {
        PairedData data = pairedData(left, right, missing);
        return pearsonChecked(data.left, data.right, data.length, alternative);
    }

    public static Result spearman(double[] left, double[] right) {
        return spearman(left, right, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static Result spearman(double[] left, double[] right, Alternative alternative) {
        return spearman(left, right, alternative, MissingPolicy.RAISE);
    }

    public static Result spearman(double[] left, double[] right, Alternative alternative, MissingPolicy missing) {
        PairedData data = pairedData(left, right, missing);
        DoubleI[] rankValues = rankScratch(data.length);
        double[] leftRanks = new double[data.length];
        double[] rightRanks = new double[data.length];
        rankAverage(data.left, data.length, rankValues, leftRanks);
        rankAverage(data.right, data.length, rankValues, rightRanks);
        double statistic = pearsonStatistic(leftRanks, rightRanks, data.length);
        if (Double.isNaN(statistic)) return new Result(Double.NaN, Double.NaN);
        double pValue = spearmanPValue(statistic, data.length, alternative);
        return new Result(statistic, pValue);
    }

    public static Result kendall(double[] left, double[] right) {
        return kendall(left, right, KendallVariant.B, KendallMethod.AUTO, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static Result kendall(double[] left, double[] right, Alternative alternative) {
        return kendall(left, right, KendallVariant.B, KendallMethod.AUTO, alternative, MissingPolicy.RAISE);
    }

    public static Result kendall(double[] left, double[] right, KendallVariant variant,
                                 KendallMethod method, Alternative alternative, MissingPolicy missing) {
        PairedData data = pairedData(left, right, missing);
        return kendallChecked(data.left, data.right, data.length, variant, method, alternative);
    }

    public static MatrixResult pearson(double[] data, int observations, int variables) {
        return pearson(data, observations, variables, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static MatrixResult pearson(double[] data, int observations, int variables,
                                       Alternative alternative, MissingPolicy missing) {
        checkMatrix(data, observations, variables);
        MissingPolicy policy = missingPolicy(missing);
        checkMatrixMissingPolicy(data, policy);
        if (observations > 2 && completeMatrixFastPath(data, policy)) {
            return pearsonCompleteMatrix(data, observations, variables, alternative);
        }
        return pairwiseChecked(data, observations, variables, alternative, policy, Correlation::pearson);
    }

    public static MatrixResult spearman(double[] data, int observations, int variables) {
        return spearman(data, observations, variables, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static MatrixResult spearman(double[] data, int observations, int variables,
                                        Alternative alternative, MissingPolicy missing) {
        checkMatrix(data, observations, variables);
        MissingPolicy policy = missingPolicy(missing);
        checkMatrixMissingPolicy(data, policy);
        if (observations > 2 && completeMatrixFastPath(data, policy)) {
            return spearmanCompleteMatrix(data, observations, variables, alternative);
        }
        return pairwiseChecked(data, observations, variables, alternative, policy, Correlation::spearman);
    }

    public static MatrixResult kendall(double[] data, int observations, int variables) {
        return kendall(data, observations, variables, Alternative.TWO_SIDED, MissingPolicy.RAISE);
    }

    public static MatrixResult kendall(double[] data, int observations, int variables,
                                       Alternative alternative, MissingPolicy missing) {
        checkMatrix(data, observations, variables);
        MissingPolicy policy = missingPolicy(missing);
        checkMatrixMissingPolicy(data, policy);
        double[][] matrixColumns = columns(data, observations, variables);
        double[] statistic = new double[variables * variables];
        double[] pValue = new double[variables * variables];
        for (int row = 0; row < variables; row++) {
            for (int column = row; column < variables; column++) {
                int offset = row * variables + column;
                if (row == column) {
                    statistic[offset] = 1.0;
                    pValue[offset] = 0.0;
                } else {
                    Result result = kendall(matrixColumns[row], matrixColumns[column],
                            KendallVariant.B, KendallMethod.AUTO, alternative, policy);
                    statistic[offset] = result.statistic;
                    pValue[offset] = result.pValue;
                }
                statistic[column * variables + row] = statistic[offset];
                pValue[column * variables + row] = pValue[offset];
            }
        }
        return new MatrixResult(statistic, pValue, variables);
    }

    private interface PairwiseFunction {
        Result compute(double[] left, double[] right, Alternative alternative, MissingPolicy missing);
    }

    private static MatrixResult pairwiseChecked(double[] data, int observations, int variables,
                                                Alternative alternative, MissingPolicy missing,
                                                PairwiseFunction function) {
        double[][] matrixColumns = columns(data, observations, variables);
        double[] statistic = new double[variables * variables];
        double[] pValue = new double[variables * variables];
        for (int row = 0; row < variables; row++) {
            double[] left = matrixColumns[row];
            for (int column = row; column < variables; column++) {
                int offset = row * variables + column;
                if (row == column) {
                    statistic[offset] = 1.0;
                    pValue[offset] = 0.0;
                } else {
                    Result result = function.compute(left, matrixColumns[column], alternative, missing);
                    statistic[offset] = result.statistic;
                    pValue[offset] = result.pValue;
                }
                statistic[column * variables + row] = statistic[offset];
                pValue[column * variables + row] = pValue[offset];
            }
        }
        return new MatrixResult(statistic, pValue, variables);
    }

    private static MatrixResult pearsonCompleteMatrix(double[] data, int observations, int variables,
                                                      Alternative alternative) {
        NormalizedMatrix normalized = normalizedColumns(data, observations, variables, false);
        return completeMatrixResult(normalized, observations, variables, alternative, true);
    }

    private static MatrixResult spearmanCompleteMatrix(double[] data, int observations, int variables,
                                                       Alternative alternative) {
        NormalizedMatrix normalized = normalizedColumns(data, observations, variables, true);
        return completeMatrixResult(normalized, observations, variables, alternative, false);
    }

    private static MatrixResult completeMatrixResult(NormalizedMatrix normalized, int observations, int variables,
                                                     Alternative alternative, boolean pearsonPValue) {
        double[] statistic = new double[variables * variables];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, variables, variables, observations,
                1.0, normalized.data, 0, observations, normalized.data, 0, observations,
                0.0, statistic, 0, variables);
        double[] pValue = new double[variables * variables];
        for (int row = 0; row < variables; row++) {
            for (int column = row; column < variables; column++) {
                int offset = row * variables + column;
                if (row == column) {
                    statistic[offset] = 1.0;
                    pValue[offset] = 0.0;
                } else if (!normalized.valid[row] || !normalized.valid[column]) {
                    statistic[offset] = Double.NaN;
                    pValue[offset] = Double.NaN;
                } else {
                    double value = clampCorrelation(statistic[offset]);
                    statistic[offset] = value;
                    pValue[offset] = pearsonPValue
                            ? pearsonPValue(value, observations, alternative)
                            : spearmanPValue(value, observations, alternative);
                }
                statistic[column * variables + row] = statistic[offset];
                pValue[column * variables + row] = pValue[offset];
            }
        }
        return new MatrixResult(statistic, pValue, variables);
    }

    private static NormalizedMatrix normalizedColumns(double[] data, int observations, int variables, boolean rank) {
        double[] normalized = new double[variables * observations];
        boolean[] valid = new boolean[variables];
        double[] work = new double[observations];
        DoubleI[] rankValues = rank ? rankScratch(observations) : null;
        double[] ranks = rank ? new double[observations] : null;
        for (int variable = 0; variable < variables; variable++) {
            BLAS.dcopy(observations, data, variable, variables, work, 0, 1);
            double[] values = work;
            if (rank) {
                rankAverage(work, observations, rankValues, ranks);
                values = ranks;
            }
            valid[variable] = normalize(values, observations, normalized, variable * observations);
        }
        return new NormalizedMatrix(normalized, valid);
    }

    private static boolean normalize(double[] values, int length, double[] out, int offset) {
        double mean = VectorOps.mean(values, 0, length);
        double max = centeredMaxAbs(values, length, mean);
        if (max == 0.0) return false;
        double scaledSquares = 0.0;
        for (int index = 0; index < length; index++) {
            double scaled = (values[index] - mean) / max;
            scaledSquares += scaled * scaled;
        }
        double norm = max * Math.sqrt(scaledSquares);
        for (int index = 0; index < length; index++) out[offset + index] = (values[index] - mean) / norm;
        return true;
    }

    private static Result pearsonChecked(double[] left, double[] right, int length, Alternative alternative) {
        if (length == 2) {
            if (left[0] == left[1] || right[0] == right[1]) return new Result(Double.NaN, Double.NaN);
            double product = (left[1] - left[0]) * (right[1] - right[0]);
            return new Result(Math.copySign(1.0, product), 1.0);
        }
        double statistic = pearsonStatistic(left, right, length);
        if (Double.isNaN(statistic)) return new Result(Double.NaN, Double.NaN);
        return new Result(statistic, pearsonPValue(statistic, length, alternative));
    }

    private static double pearsonStatistic(double[] left, double[] right, int length) {
        double leftMean = VectorOps.mean(left, 0, length);
        double rightMean = VectorOps.mean(right, 0, length);
        double leftMax = centeredMaxAbs(left, length, leftMean);
        double rightMax = centeredMaxAbs(right, length, rightMean);
        if (leftMax == 0.0 || rightMax == 0.0) return Double.NaN;

        double leftScaledSquares = 0.0;
        double rightScaledSquares = 0.0;
        for (int index = 0; index < length; index++) {
            double leftScaled = (left[index] - leftMean) / leftMax;
            double rightScaled = (right[index] - rightMean) / rightMax;
            leftScaledSquares += leftScaled * leftScaled;
            rightScaledSquares += rightScaled * rightScaled;
        }
        double leftNorm = leftMax * Math.sqrt(leftScaledSquares);
        double rightNorm = rightMax * Math.sqrt(rightScaledSquares);
        double statistic = 0.0;
        for (int index = 0; index < length; index++) {
            statistic += ((left[index] - leftMean) / leftNorm) * ((right[index] - rightMean) / rightNorm);
        }
        return clampCorrelation(statistic);
    }

    private static double pearsonPValue(double statistic, int length, Alternative alternative) {
        if (Math.abs(statistic) == 1.0) {
            return switch (alternative) {
                case LESS -> statistic < 0.0 ? 0.0 : 1.0;
                case GREATER -> statistic < 0.0 ? 1.0 : 0.0;
                case TWO_SIDED -> 0.0;
            };
        }
        double shape = length / 2.0 - 1.0;
        BetaDistribution distribution = new BetaDistribution(shape, shape);
        double transformed = unitClamp((statistic + 1.0) * 0.5);
        return switch (alternative) {
            case LESS -> distribution.cdf(transformed);
            case GREATER -> distribution.ccdf(transformed);
            case TWO_SIDED -> probabilityClamp(2.0 * distribution.cdf(unitClamp((1.0 - Math.abs(statistic)) * 0.5)));
        };
    }

    private static double spearmanPValue(double statistic, int length, Alternative alternative) {
        int degreesOfFreedom = length - 2;
        if (degreesOfFreedom <= 0) return Double.NaN;
        double denominator = Math.max(0.0, (statistic + 1.0) * (1.0 - statistic));
        double tStatistic = denominator == 0.0
                ? Math.copySign(Double.POSITIVE_INFINITY, statistic)
                : statistic * Math.sqrt(degreesOfFreedom / denominator);
        StudentsTDistribution distribution = new StudentsTDistribution(degreesOfFreedom);
        return switch (alternative) {
            case LESS -> distribution.cdf(tStatistic);
            case GREATER -> distribution.ccdf(tStatistic);
            case TWO_SIDED -> probabilityClamp(2.0 * distribution.ccdf(Math.abs(tStatistic)));
        };
    }

    private static Result kendallChecked(double[] left, double[] right, int length, KendallVariant variant,
                                         KendallMethod method, Alternative alternative) {
        RankedPairs ranked = denseRanksForKendall(left, right, length);
        long discordant = kendallDiscordant(ranked.leftRanks, ranked.rightRanks);
        TieStats jointTies = jointTieStats(ranked.leftRanks, ranked.rightRanks);
        TieStats leftTies = rankTieStats(ranked.leftRanks);
        TieStats rightTies = rankTieStats(ranked.rightRanks);
        long totalPairs = ((long) length * (length - 1)) / 2;
        if (leftTies.pairCount == totalPairs || rightTies.pairCount == totalPairs) {
            return new Result(Double.NaN, Double.NaN);
        }

        long concordantMinusDiscordant = totalPairs - leftTies.pairCount - rightTies.pairCount
                + jointTies.pairCount - 2L * discordant;
        double statistic = switch (variant) {
            case B -> concordantMinusDiscordant
                    / Math.sqrt((double) (totalPairs - leftTies.pairCount))
                    / Math.sqrt((double) (totalPairs - rightTies.pairCount));
            case C -> {
                int minClasses = Math.min(ranked.leftClassCount, ranked.rightClassCount);
                yield 2.0 * concordantMinusDiscordant
                        / (length * (double) length * (minClasses - 1.0) / minClasses);
            }
        };
        statistic = clampCorrelation(statistic);

        KendallMethod resolved = method;
        if (resolved == KendallMethod.EXACT && (leftTies.pairCount != 0 || rightTies.pairCount != 0)) {
            throw new IllegalArgumentException("Kendall exact p-value cannot be used when ties are present");
        }
        if (resolved == KendallMethod.AUTO) {
            if (leftTies.pairCount == 0 && rightTies.pairCount == 0
                    && (length <= 33 || Math.min(discordant, totalPairs - discordant) <= 1)) {
                resolved = KendallMethod.EXACT;
            } else {
                resolved = KendallMethod.ASYMPTOTIC;
            }
        }

        double pValue = resolved == KendallMethod.EXACT
                ? kendallExactPValue(length, totalPairs - discordant, alternative)
                : kendallAsymptoticPValue(length, concordantMinusDiscordant, leftTies, rightTies, alternative);
        return new Result(statistic, pValue);
    }

    private static double kendallAsymptoticPValue(int length, long concordantMinusDiscordant,
                                                  TieStats leftTies, TieStats rightTies,
                                                  Alternative alternative) {
        double samplePairsTwice = length * (length - 1.0);
        double variance = ((samplePairsTwice * (2.0 * length + 5.0) - leftTies.thirdOrder - rightTies.thirdOrder) / 18.0)
                + (2.0 * leftTies.pairCount * rightTies.pairCount) / samplePairsTwice
                + (leftTies.secondOrder * rightTies.secondOrder) / (9.0 * samplePairsTwice * (length - 2.0));
        double zStatistic = concordantMinusDiscordant / Math.sqrt(variance);
        return switch (alternative) {
            case LESS -> STANDARD_NORMAL.cdf(zStatistic);
            case GREATER -> STANDARD_NORMAL.ccdf(zStatistic);
            case TWO_SIDED -> probabilityClamp(2.0 * STANDARD_NORMAL.ccdf(Math.abs(zStatistic)));
        };
    }

    private static double kendallExactPValue(int length, long concordant, Alternative alternative) {
        long totalPairs = ((long) length * (length - 1)) / 2;
        boolean inRightTail = concordant >= totalPairs - concordant;
        boolean alternativeGreater = alternative == Alternative.GREATER;
        int inversionCount = Math.toIntExact(Math.min(concordant, totalPairs - concordant));

        double probability;
        double probabilityMassAtCount;
        if (length <= 0) {
            throw new IllegalArgumentException("length must be positive");
        } else if (inversionCount < 0 || 4L * inversionCount > (long) length * (length - 1)) {
            throw new IllegalArgumentException("invalid Kendall inversion count");
        } else if (length == 1) {
            probability = 1.0;
            probabilityMassAtCount = 1.0;
        } else if (length == 2) {
            probability = 1.0;
            probabilityMassAtCount = 0.5;
        } else if (inversionCount == 0) {
            probability = length < 171 ? 2.0 / factorial(length) : 0.0;
            probabilityMassAtCount = probability / 2.0;
        } else if (inversionCount == 1) {
            probability = length < 172 ? 2.0 / factorial(length - 1) : 0.0;
            probabilityMassAtCount = (length - 1.0) / factorial(length);
        } else if (4L * inversionCount == (long) length * (length - 1)
                && alternative == Alternative.TWO_SIDED) {
            probability = 1.0;
            probabilityMassAtCount = Double.NaN;
        } else if (length < 171) {
            double[] counts = kendallMahonianCounts(length, inversionCount, false);
            probability = 2.0 * sum(counts) / factorial(length);
            probabilityMassAtCount = counts[inversionCount] / factorial(length);
        } else {
            double[] counts = kendallMahonianCounts(length, inversionCount, true);
            probability = sum(counts);
            probabilityMassAtCount = counts[inversionCount] / 2.0;
        }

        if (alternative != Alternative.TWO_SIDED) {
            if (inRightTail == alternativeGreater) {
                probability /= 2.0;
            } else {
                probability = 1.0 - probability / 2.0 + probabilityMassAtCount;
            }
        }
        return probabilityClamp(probability);
    }

    private static double[] kendallMahonianCounts(int length, int inversionCount, boolean scaled) {
        double[] counts = new double[inversionCount + 1];
        counts[0] = 1.0;
        if (inversionCount >= 1) counts[1] = 1.0;
        for (int rank = 3; rank <= length; rank++) {
            for (int index = 1; index <= inversionCount; index++) counts[index] += counts[index - 1];
            if (scaled) {
                for (int index = 0; index <= inversionCount; index++) counts[index] /= rank;
            }
            if (rank <= inversionCount) {
                for (int index = inversionCount; index >= rank; index--) counts[index] -= counts[index - rank];
            }
        }
        return counts;
    }

    static double[] rankAverage(double[] values, int length) {
        double[] ranks = new double[length];
        rankAverage(values, length, rankScratch(length), ranks);
        return ranks;
    }

    private static void rankAverage(double[] values, int length, DoubleI[] rankValues, double[] ranks) {
        for (int index = 0; index < length; index++) rankValues[index] = new DoubleI(values[index], index);
        Arrays.sort(rankValues, 0, length, RANK_COMPARATOR);
        int start = 0;
        while (start < length) {
            int end = start + 1;
            while (end < length && Double.compare(rankValues[start].value(), rankValues[end].value()) == 0) end++;
            double averageRank = (start + 1.0 + end) * 0.5;
            for (int index = start; index < end; index++) ranks[rankValues[index]._2()] = averageRank;
            start = end;
        }
    }

    private static DoubleI[] rankScratch(int length) {
        return new DoubleI[length];
    }

    private static RankedPairs denseRanksForKendall(double[] left, double[] right, int length) {
        KendallPair[] pairs = new KendallPair[length];
        for (int index = 0; index < length; index++) pairs[index] = new KendallPair(left[index], right[index]);
        Arrays.sort(pairs, Comparator.comparingDouble(pair -> pair.right));
        int rightClassCount = assignRightRanks(pairs);
        Arrays.sort(pairs, Comparator.comparingDouble(pair -> pair.left));
        int leftClassCount = assignLeftRanks(pairs);
        int[] leftRanks = new int[length];
        int[] rightRanks = new int[length];
        for (int index = 0; index < length; index++) {
            leftRanks[index] = pairs[index].leftRank;
            rightRanks[index] = pairs[index].rightRank;
        }
        return new RankedPairs(leftRanks, rightRanks, leftClassCount, rightClassCount);
    }

    private static int assignRightRanks(KendallPair[] pairs) {
        int rank = 0;
        double previous = Double.NaN;
        for (int index = 0; index < pairs.length; index++) {
            if (index == 0 || Double.compare(pairs[index].right, previous) != 0) {
                rank++;
                previous = pairs[index].right;
            }
            pairs[index].rightRank = rank;
        }
        return rank;
    }

    private static int assignLeftRanks(KendallPair[] pairs) {
        int rank = 0;
        double previous = Double.NaN;
        for (int index = 0; index < pairs.length; index++) {
            if (index == 0 || Double.compare(pairs[index].left, previous) != 0) {
                rank++;
                previous = pairs[index].left;
            }
            pairs[index].leftRank = rank;
        }
        return rank;
    }

    private static long kendallDiscordant(int[] leftRanks, int[] rightRanks) {
        int length = leftRanks.length;
        int maxRightRank = 0;
        for (int rightRank : rightRanks) maxRightRank = Math.max(maxRightRank, rightRank);
        long[] tree = new long[maxRightRank + 1];
        long discordant = 0L;
        int groupStart = 0;
        int groupEnd = 0;
        while (groupStart < length) {
            while (groupEnd < length && leftRanks[groupStart] == leftRanks[groupEnd]) {
                discordant += groupStart;
                int queryIndex = rightRanks[groupEnd];
                while (queryIndex != 0) {
                    discordant -= tree[queryIndex];
                    queryIndex &= queryIndex - 1;
                }
                groupEnd++;
            }
            while (groupStart < groupEnd) {
                int updateIndex = rightRanks[groupStart];
                while (updateIndex <= maxRightRank) {
                    tree[updateIndex]++;
                    updateIndex += updateIndex & -updateIndex;
                }
                groupStart++;
            }
        }
        return discordant;
    }

    private static TieStats jointTieStats(int[] leftRanks, int[] rightRanks) {
        long pairCount = 0L;
        int start = 0;
        while (start < leftRanks.length) {
            int end = start + 1;
            while (end < leftRanks.length && leftRanks[start] == leftRanks[end]
                    && rightRanks[start] == rightRanks[end]) {
                end++;
            }
            long count = end - start;
            pairCount += count * (count - 1L) / 2L;
            start = end;
        }
        return new TieStats(pairCount, 0.0, 0.0);
    }

    private static TieStats rankTieStats(int[] ranks) {
        int maxRank = 0;
        for (int rank : ranks) maxRank = Math.max(maxRank, rank);
        int[] counts = new int[maxRank + 1];
        for (int rank : ranks) counts[rank]++;
        long pairCount = 0L;
        double secondOrder = 0.0;
        double thirdOrder = 0.0;
        for (int count : counts) {
            if (count > 1) {
                pairCount += count * (count - 1L) / 2L;
                secondOrder += count * (count - 1.0) * (count - 2.0);
                thirdOrder += count * (count - 1.0) * (2.0 * count + 5.0);
            }
        }
        return new TieStats(pairCount, secondOrder, thirdOrder);
    }

    private static PairedData pairedData(double[] left, double[] right, MissingPolicy missing) {
        if (left.length != right.length) throw new IllegalArgumentException("inputs must have the same length");
        if (left.length < 2) throw new IllegalArgumentException("inputs must have length at least 2");
        MissingPolicy policy = missing == null ? MissingPolicy.RAISE : missing;
        return switch (policy) {
            case RAISE -> {
                checkFinite(left, "left");
                checkFinite(right, "right");
                yield new PairedData(left, right, left.length);
            }
            case DROP -> dropMissing(left, right);
            case NONE -> new PairedData(left, right, left.length);
            case CONSERVATIVE -> throw new IllegalArgumentException("CONSERVATIVE missing policy is not supported for correlations");
        };
    }

    private static PairedData dropMissing(double[] left, double[] right) {
        int count = 0;
        for (int index = 0; index < left.length; index++) {
            if (Double.isFinite(left[index]) && Double.isFinite(right[index])) count++;
        }
        if (count < 2) throw new IllegalArgumentException("fewer than two finite paired observations");
        double[] cleanLeft = new double[count];
        double[] cleanRight = new double[count];
        int output = 0;
        for (int index = 0; index < left.length; index++) {
            if (Double.isFinite(left[index]) && Double.isFinite(right[index])) {
                cleanLeft[output] = left[index];
                cleanRight[output] = right[index];
                output++;
            }
        }
        return new PairedData(cleanLeft, cleanRight, count);
    }

    private static void checkFinite(double[] values, String name) {
        if (VectorOps.hasInf(values)) throw new IllegalArgumentException(name + " contains NaN or infinite values");
    }

    private static void checkMatrix(double[] data, int observations, int variables) {
        if (observations < 2) throw new IllegalArgumentException("observations must be at least 2");
        if (variables < 1) throw new IllegalArgumentException("variables must be positive");
        if (data.length != observations * variables) {
            throw new IllegalArgumentException("data length must equal observations * variables");
        }
    }

    private static void checkMatrixMissingPolicy(double[] data, MissingPolicy missing) {
        MissingPolicy policy = missingPolicy(missing);
        switch (policy) {
            case RAISE -> checkFinite(data, "data");
            case DROP, NONE -> {
            }
            case CONSERVATIVE -> throw new IllegalArgumentException("CONSERVATIVE missing policy is not supported for correlations");
        }
    }

    private static MissingPolicy missingPolicy(MissingPolicy missing) {
        return missing == null ? MissingPolicy.RAISE : missing;
    }

    private static boolean completeMatrixFastPath(double[] data, MissingPolicy policy) {
        return switch (policy) {
            case RAISE -> true;
            case NONE -> !VectorOps.hasInf(data);
            case DROP, CONSERVATIVE -> false;
        };
    }

    private static double[] column(double[] data, int observations, int variables, int column) {
        double[] out = new double[observations];
        BLAS.dcopy(observations, data, column, variables, out, 0, 1);
        return out;
    }

    private static double[][] columns(double[] data, int observations, int variables) {
        double[][] out = new double[variables][];
        for (int variable = 0; variable < variables; variable++) {
            out[variable] = column(data, observations, variables, variable);
        }
        return out;
    }

    private static double centeredMaxAbs(double[] values, int length, double mean) {
        double max = 0.0;
        for (int index = 0; index < length; index++) max = Math.max(max, Math.abs(values[index] - mean));
        return max;
    }

    private static double sum(double[] values) {
        double total = 0.0;
        for (double value : values) total += value;
        return total;
    }

    private static double factorial(int value) {
        double result = 1.0;
        for (int factor = 2; factor <= value; factor++) result *= factor;
        return result;
    }

    private static double unitClamp(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static double clampCorrelation(double statistic) {
        double clipped = Math.max(-1.0, Math.min(1.0, statistic));
        if (1.0 - Math.abs(clipped) <= 4.0 * Math.ulp(1.0)) {
            return Math.copySign(1.0, clipped);
        }
        return clipped;
    }

    private static double probabilityClamp(double value) {
        if (Double.isNaN(value)) return Double.NaN;
        return Math.max(0.0, Math.min(1.0, value));
    }

    private record PairedData(double[] left, double[] right, int length) {
    }

    private record NormalizedMatrix(double[] data, boolean[] valid) {
    }

    private static final class KendallPair {
        final double left;
        final double right;
        int leftRank;
        int rightRank;

        KendallPair(double left, double right) {
            this.left = left;
            this.right = right;
        }
    }

    private record RankedPairs(int[] leftRanks, int[] rightRanks, int leftClassCount, int rightClassCount) {
    }

    private record TieStats(long pairCount, double secondOrder, double thirdOrder) {
    }
}