package com.curioloop.yum4j.kalman.arena;

public value record UnivariateScratchLayout(int tmpABase,
                                      int tmpPBase,
                                      int tmpPZiBase,
                                      int tmpRQBase,
                                      int tmpKBase,
                                      int tmpFBase,
                                      int tmpPInfBase,
                                      int tmpPStarBase,
                                      int tmpTPBase,
                                      int tmpMStarBase,
                                      int tmpMInfBase,
                                      int tmpK0Base,
                                      int tmpK1Base,
                                      int totalLength) {

    public static UnivariateScratchLayout create(int kEndog, int kStates) {
        int ksKs = kStates * kStates;
        int ksKe = kStates * kEndog;
        int keKe = kEndog * kEndog;
        int offset = 0;

        int tmpABase = offset;
        offset += kStates;
        int tmpPBase = offset;
        offset += ksKs;
        int tmpPZiBase = offset;
        offset += kStates;
        int tmpRQBase = offset;
        offset += ksKs;
        int tmpKBase = offset;
        offset += ksKe;
        int tmpFBase = offset;
        offset += keKe;
        int tmpPInfBase = offset;
        offset += ksKs;
        int tmpPStarBase = offset;
        offset += ksKs;
        int tmpTPBase = offset;
        offset += ksKs;
        int tmpMStarBase = offset;
        offset += kStates;
        int tmpMInfBase = offset;
        offset += kStates;
        int tmpK0Base = offset;
        offset += kStates;
        int tmpK1Base = offset;
        offset += kStates;

        return new UnivariateScratchLayout(
            tmpABase,
            tmpPBase,
            tmpPZiBase,
            tmpRQBase,
            tmpKBase,
            tmpFBase,
            tmpPInfBase,
            tmpPStarBase,
            tmpTPBase,
            tmpMStarBase,
            tmpMInfBase,
            tmpK0Base,
            tmpK1Base,
            offset);
    }
}