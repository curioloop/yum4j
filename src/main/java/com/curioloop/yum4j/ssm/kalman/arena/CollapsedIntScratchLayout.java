package com.curioloop.yum4j.ssm.kalman.arena;

public value record CollapsedIntScratchLayout(int selectedIndexBase,
                                              int pivotsBase,
                                              int totalLength) {

    public static CollapsedIntScratchLayout create(int kEndog) {
        int offset = 0;
        int selectedIndexBase = offset;
        offset += kEndog;
        int pivotsBase = offset;
        offset += kEndog;
        return new CollapsedIntScratchLayout(selectedIndexBase, pivotsBase, offset);
    }
}
