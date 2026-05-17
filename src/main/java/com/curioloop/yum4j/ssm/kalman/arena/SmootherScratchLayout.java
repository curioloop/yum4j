package com.curioloop.yum4j.ssm.kalman.arena;

public value record SmootherScratchLayout(int rBase,
                                    int rPrevBase,
                                    int nBase,
                                    int nPrevBase,
                                    int lBase,
                                    int tmpNpBase,
                                    int tmpRqBase,
                                    int tmpFivBase,
                                    int tmpFiZBase,
                                    int tmpFcopyBase,
                                    int tmpTkBase,
                                    int tmpKhBase,
                                    int tmpFiHBase,
                                    int tmpJBase,
                                    int selectedDesignBase,
                                    int selectedObsCovBase,
                                    int tmpKgBase,
                                    int rInfBase,
                                    int rInfPrevBase,
                                    int nInf1Base,
                                    int nInf1PrevBase,
                                    int nInf2Base,
                                    int nInf2PrevBase,
                                    int l0Base,
                                    int l1Base,
                                    int k0Base,
                                    int k1Base,
                                    int tmpMInfBase,
                                    int tmpMStarBase,
                                    int tmpTtNBase,
                                    int tmpTtNInf1Base,
                                    int tmpTtNInf2Base,
                                    int tmpPNInf1Base,
                                    int tmpPInfNInf2Base,
                                    int scratchSmoothingErrorBase,
                                    int totalLength) {

    public static SmootherScratchLayout create(int kEndog,
                                               int kStates,
                                               boolean needObservationFactor,
                                               boolean needForecastFactor,
                                               boolean needTransitionGain,
                                               boolean needDisturbanceCovariance,
                                               boolean needFactorScratch,
                                               boolean needDiffuseScratch,
                                               boolean needBorrowedSmoothingError) {
        int ksKs = kStates * kStates;
        int ksKe = kStates * kEndog;
        int keKe = kEndog * kEndog;
        int offset = 0;

        int rBase = offset;
        offset += kStates;
        int rPrevBase = offset;
        offset += kStates;
        int nBase = offset;
        offset += ksKs;
        int nPrevBase = offset;
        offset += ksKs;
        int lBase = offset;
        offset += ksKs;
        int tmpNpBase = offset;
        offset += ksKs;
        int tmpRqBase = offset;
        offset += ksKs;

        int tmpFivBase = needObservationFactor ? offset : -1;
        if (needObservationFactor) {
            offset += kEndog;
        }
        int tmpFiZBase = needObservationFactor ? offset : -1;
        if (needObservationFactor) {
            offset += ksKe;
        }

        int tmpFcopyBase = needForecastFactor ? offset : -1;
        if (needForecastFactor) {
            offset += keKe;
        }

        int tmpTkBase = -1;
        int tmpFiHBase = -1;
        if (needTransitionGain && needDisturbanceCovariance) {
            tmpTkBase = offset;
            tmpFiHBase = offset;
            offset += Math.max(ksKe, keKe);
        } else if (needTransitionGain) {
            tmpTkBase = offset;
            offset += ksKe;
        }
        int tmpKhBase = needDisturbanceCovariance ? offset : -1;
        if (needDisturbanceCovariance) {
            offset += ksKe;
        }
        if (needDisturbanceCovariance && tmpFiHBase < 0) {
            tmpFiHBase = offset;
            offset += keKe;
        }

        int tmpJBase = needFactorScratch ? offset : -1;
        if (needFactorScratch) {
            offset += ksKs;
        }

        int selectedDesignBase = needObservationFactor ? offset : -1;
        if (needObservationFactor) {
            offset += ksKe;
        }
        int selectedObsCovBase = needObservationFactor ? offset : -1;
        if (needObservationFactor) {
            offset += keKe;
        }
        int tmpKgBase = needObservationFactor ? offset : -1;
        if (needObservationFactor) {
            offset += ksKe;
        }

        int rInfBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int rInfPrevBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int nInf1Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int nInf1PrevBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int nInf2Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int nInf2PrevBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int l0Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int l1Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int k0Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int k1Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int tmpMInfBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int tmpMStarBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += kStates;
        }
        int tmpTtNBase = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int tmpTtNInf1Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int tmpTtNInf2Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int tmpPNInf1Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }
        int tmpPInfNInf2Base = needDiffuseScratch ? offset : -1;
        if (needDiffuseScratch) {
            offset += ksKs;
        }

        int scratchSmoothingErrorBase = needBorrowedSmoothingError ? offset : -1;
        if (needBorrowedSmoothingError) {
            offset += kEndog;
        }

        return new SmootherScratchLayout(
            rBase,
            rPrevBase,
            nBase,
            nPrevBase,
            lBase,
            tmpNpBase,
            tmpRqBase,
            tmpFivBase,
            tmpFiZBase,
            tmpFcopyBase,
            tmpTkBase,
            tmpKhBase,
            tmpFiHBase,
            tmpJBase,
            selectedDesignBase,
            selectedObsCovBase,
            tmpKgBase,
            rInfBase,
            rInfPrevBase,
            nInf1Base,
            nInf1PrevBase,
            nInf2Base,
            nInf2PrevBase,
            l0Base,
            l1Base,
            k0Base,
            k1Base,
            tmpMInfBase,
            tmpMStarBase,
            tmpTtNBase,
            tmpTtNInf1Base,
            tmpTtNInf2Base,
            tmpPNInf1Base,
            tmpPInfNInf2Base,
            scratchSmoothingErrorBase,
            offset);
    }
}