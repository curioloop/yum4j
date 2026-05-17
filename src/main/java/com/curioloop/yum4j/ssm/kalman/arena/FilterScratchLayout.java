package com.curioloop.yum4j.ssm.kalman.arena;

public value record FilterScratchLayout(int tmpKBase,
                                  int tmpPZtBase,
                                  int tmpFcopyBase,
                                  int tmpTPBase,
                                  int tmpVBase,
                                  int tmpRQBase,
                                  int rollingPredictedStateBase,
                                  int rollingPredictedStateCovBase,
                                  int rollingPredictedDiffuseStateCovBase,
                                  int scratchFilteredStateBase,
                                  int scratchFilteredStateCovBase,
                                  int totalLength) {

    public static FilterScratchLayout create(int scalarWidth,
                                             int kEndog,
                                             int kStates,
                                             int kPosdef,
                                             boolean needPredictedScratch,
                                             boolean needFilteredScratch,
                                             boolean needDiffusePredictedScratch) {
        int ksKe = scalarWidth * kStates * kEndog;
        int ksKs = scalarWidth * kStates * kStates;
        int keKe = scalarWidth * kEndog * kEndog;
        int updateLength = ksKe + ksKe + keKe + scalarWidth * kEndog;
        int predictionLength = Math.max(ksKs, scalarWidth * kStates * kPosdef);
        int offset = 0;

        int tmpKBase = offset;
        offset += ksKe;
        int tmpPZtBase = offset;
        offset += ksKe;
        int tmpFcopyBase = offset;
        offset += keKe;
        int tmpVBase = offset;
        offset += scalarWidth * kEndog;
        offset = Math.max(updateLength, predictionLength);
        int tmpTPBase = 0;
        int tmpRQBase = 0;
        int rollingPredictedStateBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += scalarWidth * kStates * 2;
        }
        int rollingPredictedStateCovBase = needPredictedScratch ? offset : -1;
        if (needPredictedScratch) {
            offset += ksKs * 2;
        }
        int rollingPredictedDiffuseStateCovBase = needDiffusePredictedScratch ? offset : -1;
        if (needDiffusePredictedScratch) {
            offset += ksKs * 2;
        }
        int scratchFilteredStateBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += scalarWidth * kStates;
        }
        int scratchFilteredStateCovBase = needFilteredScratch ? offset : -1;
        if (needFilteredScratch) {
            offset += ksKs;
        }

        return new FilterScratchLayout(
            tmpKBase,
            tmpPZtBase,
            tmpFcopyBase,
            tmpTPBase,
            tmpVBase,
            tmpRQBase,
            rollingPredictedStateBase,
            rollingPredictedStateCovBase,
            rollingPredictedDiffuseStateCovBase,
            scratchFilteredStateBase,
            scratchFilteredStateCovBase,
            offset);
    }
}