import evaluation.plots.ev_false_positives as false_positives
import evaluation.plots.ev_metric_vs_comparison_based as metric_vs_comparison_based
import evaluation.plots.ev_runtime as runtime
import evaluation.plots.ev_spec_complexity as spec_complexity
import evaluation.plots.ev_specs_vs_removed_cross_edges as specs_vs_removed_cross_edges
import evaluation.plots.ev_case_study_blinker as case_study_blinker


def start_evaluation(option):
    if option == 1:
        spec_ratio_vs_removed_cross_edges_evaluation()
        #
    elif option == 2:
        metric_vs_comparison_based_evaluation()
        #
    elif option == 3:
        runtime_evalutation()
        #
    elif option == 4:
        spec_complexity_evaluation()
        #
    elif option == 5:
        false_positive_evaluation()
        #
    elif option == 6:
        case_study_blinker_evaluation()
        #
    else:
        print("Please choose an option between 1 and 6 to start an evaluation.")


def spec_ratio_vs_removed_cross_edges_evaluation():
    specs_vs_removed_cross_edges.start()


def metric_vs_comparison_based_evaluation():
    metric_vs_comparison_based.start()


def runtime_evalutation():
    runtime.start()


def spec_complexity_evaluation():
    spec_complexity.start()


def false_positive_evaluation():
    false_positives.start()


def case_study_blinker_evaluation():
    case_study_blinker.start()
