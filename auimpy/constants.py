DEP_VARS = ['rt', 'er', 'drift_rate',
            'boundary_separation', 'non_decision_time']
MODEL_COLS = DEP_VARS[2:]

TITALIZED_DEP_VARS = ['Mean Response Time', 'Error Rate', 'Drift Rate',
                      'Boundary Separation', 'Non-Decision Time']
UNITS = ["ms", "%", 'v', 'a', 's']
DEP_VARS_LABELS = ['{} ({})'.format(i, m) for (i, m) in zip(
    TITALIZED_DEP_VARS, UNITS)]
DEP_VAR_LABEL_DICT = dict(zip(DEP_VARS, DEP_VARS_LABELS))
COLORS = ['dodgerblue', 'darkred', 'orange']
TITALIZED_EFFECT_LABELS = dict(zip(
    ['condition', 'cuenumber', 'condition:cuenumber'],
    ['experimental condition', 'response finger', 'the interaction term']
))
SLOW_RT = 950
FAST_RT = 150
CONDS = ['Compatible', 'Incompatible', 'Irrelevant']
