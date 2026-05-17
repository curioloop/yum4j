package com.curioloop.yum4j.ssm.kalman.arena;

public value record ZFilterScratchLayout(int mode,
                                   int tmpABase,
                                   int tmpKBase,
                                   int tmpPZtBase,
                                   int tmpFcopyBase,
                                   int tmpTPBase,
                                   int tmpVBase,
                                   int tmpRQBase,
                                   int tmpPInfBase,
                                   int tmpPStarBase,
                                   int tmpMStarBase,
                                   int tmpMInfBase,
                                   int tmpK0Base,
                                   int tmpK1Base,
                                   int rollingPredictedStateBase,
                                   int rollingPredictedStateCovBase,
                                   int rollingPredictedDiffuseStateCovBase,
                                   int scratchFilteredStateBase,
                                   int scratchFilteredStateCovBase,
                                   int totalLength) {

    public static final int MODE_CONVENTIONAL = 0;
    public static final int MODE_UNIVARIATE = 1;
    public static final int MODE_EXACT_DIFFUSE = 2;

    public static ZFilterScratchLayout create(int kEndog,
                                              int kStates,
                                              int kPosdef,
                                              boolean needPredictedScratch,
                                              boolean needFilteredScratch,
                                              boolean needDiffusePredictedScratch) {
        return createConventional(kEndog, kStates, kPosdef,
            needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch);
    }

    public static ZFilterScratchLayout createConventional(int kEndog,
                                                          int kStates,
                                                          int kPosdef,
                                                          boolean needPredictedScratch,
                                                          boolean needFilteredScratch,
                                                          boolean needDiffusePredictedScratch) {
        int ksKe = 2 * kStates * kEndog;
        int ksKs = 2 * kStates * kStates;
        int keKe = 2 * kEndog * kEndog;
        int updateLength = ksKe + ksKe + keKe + 2 * kEndog;
        int predictionLength = Math.max(ksKs, 2 * kStates * kPosdef);
        int offset = 0;

        int tmpKBase = offset;
        offset += ksKe;
        int tmpPZtBase = offset;
        offset += ksKe;
        int tmpFcopyBase = offset;
        offset += keKe;
        int tmpVBase = offset;
        offset += 2 * kEndog;
        offset = Math.max(updateLength, predictionLength);
        int tmpTPBase = 0;
        int tmpRQBase = 0;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 4 * kStates;
        }
        int rollingPredictedStateCovBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * ksKs;
        }
        int rollingPredictedDiffuseStateCovBase = needDiffusePredictedScratch ? offset : -1;
        if (needDiffusePredictedScratch) {
            offset += 2 * ksKs;
        }
        int scratchFilteredStateBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += 2 * kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new ZFilterScratchLayout(
            MODE_CONVENTIONAL,
            -1,
            tmpKBase,
            tmpPZtBase,
            tmpFcopyBase,
            tmpTPBase,
            tmpVBase,
            tmpRQBase,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            rollingPredictedDiffuseStateCovBase,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }

    public static ZFilterScratchLayout createUnivariate(int kEndog,
                                                        int kStates,
                                                        int kPosdef,
                                                        boolean needPredictedScratch,
                                                        boolean needFilteredScratch) {
        int ksKs = 2 * kStates * kStates;
        int phaseLength = Math.max(2 * kStates, Math.max(ksKs, 2 * kStates * kPosdef));
        int offset = 0;

        int tmpABase = offset;
        offset += 2 * kStates;
        int tmpPStarBase = offset;
        offset += ksKs;
        int phaseBase = offset;
        offset += phaseLength;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 4 * kStates;
        }
        int rollingPredictedStateCovBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * ksKs;
        }
        int scratchFilteredStateBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += 2 * kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new ZFilterScratchLayout(
            MODE_UNIVARIATE,
            tmpABase,
            -1,
            -1,
            -1,
            phaseBase,
            -1,
            phaseBase,
            -1,
            tmpPStarBase,
            phaseBase,
            -1,
            -1,
            -1,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            -1,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }

    public static ZFilterScratchLayout createExactDiffuse(int kEndog,
                                                          int kStates,
                                                          int kPosdef,
                                                          boolean needPredictedScratch,
                                                          boolean needFilteredScratch,
                                                          boolean needDiffusePredictedScratch) {
        int ksKs = 2 * kStates * kStates;
        int phaseLength = Math.max(8 * kStates, Math.max(ksKs, 2 * kStates * kPosdef));
        int offset = 0;

        int tmpABase = offset;
        offset += 2 * kStates;
        int tmpPInfBase = offset;
        offset += ksKs;
        int tmpPStarBase = offset;
        offset += ksKs;
        int phaseBase = offset;
        offset += phaseLength;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 4 * kStates;
        }
        int rollingPredictedStateCovBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * ksKs;
        }
        int rollingPredictedDiffuseStateCovBase = needDiffusePredictedScratch ? offset : -1;
        if (needDiffusePredictedScratch) {
            offset += 2 * ksKs;
        }
        int scratchFilteredStateBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += 2 * kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new ZFilterScratchLayout(
            MODE_EXACT_DIFFUSE,
            tmpABase,
            -1,
            -1,
            -1,
            phaseBase,
            -1,
            phaseBase,
            tmpPInfBase,
            tmpPStarBase,
            phaseBase,
            phaseBase + 2 * kStates,
            phaseBase + 4 * kStates,
            phaseBase + 6 * kStates,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            rollingPredictedDiffuseStateCovBase,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }
}