package com.curioloop.yum4j.ssm.kalman.arena;

public value record UnivariateScratchLayout(int mode,
                                      int tmpABase,
                                      int tmpPBase,
                                      int tmpPZiBase,
                                      int tmpRQBase,
                                      int tmpPInfBase,
                                      int tmpPStarBase,
                                      int tmpTPBase,
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

    public static final int MODE_REGULAR = 0;
    public static final int MODE_EXACT_DIFFUSE = 1;

    public static UnivariateScratchLayout create(int kEndog, int kStates) {
        return createRegular(kEndog, kStates, kStates, false, false);
    }

    public static UnivariateScratchLayout create(int kEndog,
                                                int kStates,
                                                int kPosdef,
                                                boolean exactDiffuse,
                                                boolean needPredictedScratch,
                                                boolean needFilteredScratch,
                                                boolean needDiffusePredictedScratch) {
        return exactDiffuse
            ? createExactDiffuse(kEndog, kStates, kPosdef,
                needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch)
            : createRegular(kEndog, kStates, kPosdef,
                needPredictedScratch, needFilteredScratch);
    }

    public static UnivariateScratchLayout createRegular(int kEndog,
                                                       int kStates,
                                                       int kPosdef,
                                                       boolean needPredictedScratch,
                                                       boolean needFilteredScratch) {
        int ksKs = kStates * kStates;
        int phaseLength = Math.max(kStates, Math.max(ksKs, kStates * kPosdef));
        int offset = 0;

        int tmpABase = offset;
        offset += kStates;
        int tmpPBase = offset;
        offset += ksKs;
        int phaseBase = offset;
        offset += phaseLength;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * kStates;
        }
        int rollingPredictedStateCovBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * ksKs;
        }
        int scratchFilteredStateBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new UnivariateScratchLayout(
            MODE_REGULAR,
            tmpABase,
            tmpPBase,
            phaseBase,
            phaseBase,
            -1,
            -1,
            -1,
            -1,
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

    public static UnivariateScratchLayout createExactDiffuse(int kEndog,
                                                            int kStates,
                                                            int kPosdef,
                                                            boolean needPredictedScratch,
                                                            boolean needFilteredScratch,
                                                            boolean needDiffusePredictedScratch) {
        int ksKs = kStates * kStates;
        int phaseLength = Math.max(4 * kStates, Math.max(ksKs, kStates * kPosdef));
        int offset = 0;

        int tmpABase = offset;
        offset += kStates;
        int tmpPInfBase = offset;
        offset += ksKs;
        int tmpPStarBase = offset;
        offset += ksKs;
        int phaseBase = offset;
        offset += phaseLength;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += 2 * kStates;
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
            offset += kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new UnivariateScratchLayout(
            MODE_EXACT_DIFFUSE,
            tmpABase,
            -1,
            -1,
            phaseBase,
            tmpPInfBase,
            tmpPStarBase,
            phaseBase,
            phaseBase,
            phaseBase + kStates,
            phaseBase + 2 * kStates,
            phaseBase + 3 * kStates,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            rollingPredictedDiffuseStateCovBase,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }
}