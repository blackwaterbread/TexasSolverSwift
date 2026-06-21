# Shared core (Qt-Core-only) solver sources for the console + serializer targets.
QT = core
CONFIG += console c++17
CONFIG -= app_bundle
TEMPLATE = app

ROOT = $$PWD/../..
INCLUDEPATH += $$ROOT

win32-g++: {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
}
linux: {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
}
QMAKE_CXXFLAGS_RELEASE *= -O2

SOURCES += \
    $$ROOT/src/library.cpp \
    $$ROOT/src/Deck.cpp \
    $$ROOT/src/Card.cpp \
    $$ROOT/src/GameTree.cpp \
    $$ROOT/src/compairer/Dic5Compairer.cpp \
    $$ROOT/src/nodes/ActionNode.cpp \
    $$ROOT/src/nodes/ChanceNode.cpp \
    $$ROOT/src/nodes/GameActions.cpp \
    $$ROOT/src/nodes/GameTreeNode.cpp \
    $$ROOT/src/nodes/ShowdownNode.cpp \
    $$ROOT/src/nodes/TerminalNode.cpp \
    $$ROOT/src/ranges/PrivateCards.cpp \
    $$ROOT/src/ranges/PrivateCardsManager.cpp \
    $$ROOT/src/ranges/RiverCombs.cpp \
    $$ROOT/src/ranges/RiverRangeManager.cpp \
    $$ROOT/src/runtime/PokerSolver.cpp \
    $$ROOT/src/solver/BestResponse.cpp \
    $$ROOT/src/solver/CfrSolver.cpp \
    $$ROOT/src/solver/PCfrSolver.cpp \
    $$ROOT/src/solver/Solver.cpp \
    $$ROOT/src/tools/CommandLineTool.cpp \
    $$ROOT/src/tools/GameTreeBuildingSettings.cpp \
    $$ROOT/src/tools/lookup8.cpp \
    $$ROOT/src/tools/PrivateRangeConverter.cpp \
    $$ROOT/src/tools/progressbar.cpp \
    $$ROOT/src/tools/Rule.cpp \
    $$ROOT/src/tools/StreetSetting.cpp \
    $$ROOT/src/tools/utils.cpp \
    $$ROOT/src/trainable/CfrPlusTrainable.cpp \
    $$ROOT/src/trainable/DiscountedCfrTrainable.cpp \
    $$ROOT/src/trainable/DiscountedCfrTrainableHF.cpp \
    $$ROOT/src/trainable/DiscountedCfrTrainableSF.cpp \
    $$ROOT/src/trainable/Trainable.cpp
