import pandas as pd

import analyzeModel


def makePredsPerformanceTable(preds_f, phase = None):
    preds = pd.read_csv(preds_f)
    perf = analyzeModel.performanceMetrics(preds)
    if phase is not None:
        perf = analyzeModel.performanceMetricsWithPhase(preds)
        perf = perf[phase]


    for k in perf.keys():
        perf[k].pop('class_counts', None)

    return pd.DataFrame(perf)
