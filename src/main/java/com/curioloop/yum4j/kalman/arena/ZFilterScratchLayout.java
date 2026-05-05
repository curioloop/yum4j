package com.curioloop.yum4j.kalman.arena;

public value record ZFilterScratchLayout(int tmpABase,
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

    public static ZFilterScratchLayout create(int kEndog,
                                              int kStates,
                                              int kPosdef,
                                              boolean needPredictedScratch,
                                              boolean needFilteredScratch,
                                              boolean needDiffusePredictedScratch) {
        int ksKe = 2 * kStates * kEndog;
        int ksKs = 2 * kStates * kStates;
        int keKe = 2 * kEndog * kEndog;
        int offset = 0;

        int tmpABase = offset;
        offset += 2 * kStates;
        int tmpKBase = offset;
        offset += ksKe;
        int tmpPZtBase = offset;
        offset += ksKe;
        int tmpFcopyBase = offset;
        offset += keKe;
        int tmpTPBase = offset;
        offset += ksKs;
        int tmpVBase = offset;
        offset += 2 * kEndog;
        int tmpRQBase = offset;
        offset += 2 * kStates * kPosdef;
        int tmpPInfBase = offset;
        offset += ksKs;
        int tmpPStarBase = offset;
        offset += ksKs;
        int tmpMStarBase = offset;
        offset += 2 * kStates;
        int tmpMInfBase = offset;
        offset += 2 * kStates;
        int tmpK0Base = offset;
        offset += 2 * kStates;
        int tmpK1Base = offset;
        offset += 2 * kStates;
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
            tmpABase,
            tmpKBase,
            tmpPZtBase,
            tmpFcopyBase,
            tmpTPBase,
            tmpVBase,
            tmpRQBase,
            tmpPInfBase,
            tmpPStarBase,
            tmpMStarBase,
            tmpMInfBase,
            tmpK0Base,
            tmpK1Base,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            rollingPredictedDiffuseStateCovBase,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }
}