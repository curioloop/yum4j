package com.curioloop.yum4j.ssm.kalman.arena;

public value record ChandrasekharScratchLayout(int currentStateBase,
                                               int currentCovBase,
                                               int filteredStateBase,
                                               int filteredCovBase,
                                               int nextStateBase,
                                               int nextCovBase,
                                               int forecastMeanBase,
                                               int forecastErrorBase,
                                               int forecastCovBase,
                                               int forecastCovWorkBase,
                                               int forecastErrorSolvedBase,
                                               int predictedDesignTransposedBase,
                                               int solvedDesignPredictedBase,
                                               int factorMWBase,
                                               int forecastInverseDesignBase,
                                               int previousForecastInverseDesignBase,
                                               int kalmanGainBase,
                                               int factorMBase,
                                               int factorWBase,
                                               int factorWNextBase,
                                               int tmpMBase,
                                               int tmpMWZBase,
                                               int transitionMinusKZBase,
                                               int luInverseWorkBase,
                                               int totalLength) {

    public static ChandrasekharScratchLayout create(int kEndog,
                                                    int kStates,
                                                    boolean needFilteredCov,
                                                    int luInverseWorkLength) {
        int keKe = kEndog * kEndog;
        int ksKs = kStates * kStates;
        int ksKe = kStates * kEndog;
        int offset = 0;

        int currentStateBase = offset;
        offset += kStates;
        int currentCovBase = offset;
        offset += ksKs;
        int filteredStateBase = offset;
        offset += kStates;
        int filteredCovBase = needFilteredCov ? offset : -1;
        if (needFilteredCov) {
            offset += ksKs;
        }
        int nextStateBase = offset;
        offset += kStates;
        int nextCovBase = offset;
        offset += ksKs;
        int forecastMeanBase = offset;
        offset += kEndog;
        int forecastErrorBase = offset;
        offset += kEndog;
        int forecastCovBase = offset;
        offset += keKe;
        int forecastCovWorkBase = offset;
        offset += keKe;
        int forecastErrorSolvedBase = offset;
        offset += kEndog;
        int predictedDesignTransposedBase = offset;
        offset += ksKe;
        int solvedDesignPredictedBase = offset;
        offset += ksKe;
        int forecastInverseDesignBase = offset;
        offset += ksKe;
        int previousForecastInverseDesignBase = offset;
        offset += ksKe;
        int kalmanGainBase = offset;
        offset += ksKe;
        int factorMBase = offset;
        offset += keKe;
        int factorWBase = offset;
        offset += ksKe;
        int factorWNextBase = offset;
        offset += ksKe;
        int tmpMBase = offset;
        offset += keKe;
        int tmpMWZBase = offset;
        offset += keKe;
        int luInverseWorkBase = luInverseWorkLength > 0 ? offset : -1;
        if (luInverseWorkLength > 0) {
            offset += luInverseWorkLength;
        }

        return new ChandrasekharScratchLayout(
            currentStateBase,
            currentCovBase,
            filteredStateBase,
            filteredCovBase,
            nextStateBase,
            nextCovBase,
            forecastMeanBase,
            forecastErrorBase,
            forecastCovBase,
            forecastCovWorkBase,
            forecastErrorSolvedBase,
            predictedDesignTransposedBase,
            solvedDesignPredictedBase,
            solvedDesignPredictedBase,
            forecastInverseDesignBase,
            previousForecastInverseDesignBase,
            kalmanGainBase,
            factorMBase,
            factorWBase,
            factorWNextBase,
            tmpMBase,
            tmpMWZBase,
            nextCovBase,
            luInverseWorkBase,
            offset);
    }

    public int standardizedWorkBase() {
        return forecastCovWorkBase;
    }
}
