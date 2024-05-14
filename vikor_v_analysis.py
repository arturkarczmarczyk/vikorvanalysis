import numpy as np
import pandas as pd
from comparator.comparison import Comparison
from pyrepo_mcda.weighting_methods import critic_weighting, entropy_weighting, std_weighting, gini_weighting
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from vikor_evaluator import VikorEvaluator

DECISION_PROBLEM = 'wfarms'

V_MIN = 0
V_MAX = 1
V_STEP = 0.02
BOOSTED_WEIGHT = 0.1

IS_DRAW_WEIGHTS_PLOTS = False
IS_RUN_CORRELATION_HEATMAPS = False
IS_RUN_SENSITIVITY_V = False
IS_RUN_SENSITIVITY_WEIGHTS = False
IS_DRAW_WEIGHTS_SENSITIVITY_PLOTS = False
IS_RUN_CLUSTERS = True

data = pd.read_csv('wind_farms_data.csv', index_col=0)

impacts = data.iloc[-1]
# print(impacts)

data = data.iloc[:-1]
# print(data)

weights_scenarios = {
    'eq': np.full(data.shape[1], 1/data.shape[1]),
    'crit': critic_weighting(data.to_numpy()),
    'ent': entropy_weighting(data.to_numpy()),
    'gini': gini_weighting(data.to_numpy()),
}

if (IS_DRAW_WEIGHTS_PLOTS):
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Computed weights of criteria for varied scenarios')
    i = 0
    for scenario, weights in weights_scenarios.items():
        i+= 1
        plt.subplot(len(weights_scenarios), 1, i)
        plt.gca().set_ylim([0, 0.6])
        plt.title(scenario)
        plt.plot(data.columns, weights, label=scenario)
        plt.grid(color='whitesmoke', linestyle='solid')
    # plt.legend()
    plt.savefig('var/weights.png')
    # plt.show()
    plt.clf()

    weights_scenarios_pd = pd.DataFrame(weights_scenarios)
    weights_scenarios_pd.to_csv('var/weights_scenarios.csv')


v_step_rounding = len(f"{V_STEP}") - 2
comparisons_v = {}

# iterate over the scenarios
for scenario, weights in weights_scenarios.items():
    print(scenario)
    # print(weights)

    comparison_v = Comparison(data.shape[0], data.shape[1])
    comparison_v.add_decision_problem(DECISION_PROBLEM, data, impacts)
    comparison_v.add_weights_set(scenario, weights)

    for v in np.arange(V_MIN, V_MAX + (V_STEP/10), V_STEP).round(v_step_rounding):
        eval = VikorEvaluator(v)
        comparison_v.add_evaluator(f'v_{v}', eval)

    comparison_v.compute()

    comparisons_v[scenario] = comparison_v

# draw heatmaps of correlations between rankings
if IS_RUN_CORRELATION_HEATMAPS:
    for scenario, comparison_v in comparisons_v.items():
        print(scenario)
        plt = comparison_v.plot_correlations_heatmap(figure_size=(50, 20))
        plt.savefig(f'var/correlation_{scenario}.png')
        # plt.show()


# sensitivity analysis based on V
if IS_RUN_SENSITIVITY_V:
    ZOOM_MAX = 2
    for scenario, comparison_v in comparisons_v.items():
        x_v_values = np.arange(V_MIN, V_MAX + (V_STEP/10), V_STEP).round(v_step_rounding)
        y_alternatives_scores = pd.DataFrame(comparison_v.scores[DECISION_PROBLEM][scenario])
        y_alternatives_ranks = pd.DataFrame(rankdata(y_alternatives_scores, axis=0, method='min'),
                                            index=y_alternatives_scores.index)

        plt.figure(figsize=(30, 10))
        plt.suptitle(f'Sensitivity analysis for scenario {scenario}')

        plt.subplot(1, 3, 1)
        plt.grid(color='whitesmoke', linestyle='solid')
        plt.title('Scores')
        plt.axvline(x=0.5, color='b', linestyle='--')
        plt.gca().invert_yaxis()
        for i in y_alternatives_scores.index:
            plt.plot(x_v_values, y_alternatives_scores.loc[i], label=i)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.grid(color='whitesmoke', linestyle='solid')
        plt.title('Ranks')
        plt.axvline(x=0.5, color='b', linestyle='--')
        plt.gca().invert_yaxis()
        for i in y_alternatives_ranks.index:
            plt.plot(x_v_values, y_alternatives_ranks.loc[i], label=i)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.grid(color='whitesmoke', linestyle='solid')
        plt.title('Scores Zoomed')
        plt.gca().invert_yaxis()
        for i in y_alternatives_scores.index:
            plt.plot(x_v_values[:ZOOM_MAX], y_alternatives_scores.loc[i][:ZOOM_MAX], label=i)
        plt.legend()

        plt.savefig(f'var/sensitivity_{scenario}.png')
        # plt.show()

        plt.clf()


# sensitivity analysis based on V
if IS_RUN_SENSITIVITY_WEIGHTS:
    weights_scenarios_weights = {}
    num_criteria = len(data.columns)

    # whatever is left to distribute among the non-boosted criteria
    reduced_weight = (1 - BOOSTED_WEIGHT) / (num_criteria - 1)

    i = 0
    weights_scenarios_weights['eq'] = np.full(num_criteria, 1 / num_criteria)
    for criterion in data.columns:
        weights_scenarios_weights[criterion] = np.full(num_criteria, reduced_weight)
        weights_scenarios_weights[criterion][i] = BOOSTED_WEIGHT
        i += 1

    if (IS_DRAW_WEIGHTS_SENSITIVITY_PLOTS):
        plt.figure(figsize=(15, 30))
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(f'Computed weights of criteria for weights sensitivity scenarios')
        i = 0
        for scenario, weights in weights_scenarios_weights.items():
            i+= 1
            plt.subplot(len(weights_scenarios_weights), 1, i)
            plt.gca().set_ylim([0, 0.1])
            plt.title(scenario)
            plt.plot(data.columns, weights, label=scenario)
            plt.grid(color='whitesmoke', linestyle='solid')
        # plt.legend()
        plt.savefig('var/weights_sensitivity.png')
        # plt.show()
        plt.clf()

    comparison_w = Comparison(data.shape[0], data.shape[1])

    comparison_w.add_decision_problem(DECISION_PROBLEM, data, impacts)

    v = 0.5
    eval = VikorEvaluator(v)
    comparison_w.add_evaluator(f'v_{v}', eval)

    for scenario, weights in weights_scenarios_weights.items():
        print(scenario)
        print(weights)
        comparison_w.add_weights_set(scenario, weights)

    comparison_w.compute()
    # plt = comparison_w.plot_correlations_heatmap(figure_size=(50, 20))
    # plt.savefig(f'var/correlation_weights.png')
    # plt.show()
    plt.clf()


    comparison_w_df = comparison_w.to_dataframe(normalize_scores=False)

    plt.clf()
    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(hspace=0.5)
    x_values = data.index.to_list()
    y_scores_eq = comparison_w_df[comparison_w_df['weights_set'] == 'eq'].iloc[0][3:]
    i = 0
    for index, row in comparison_w_df.iterrows():
        scenario = row['weights_set']
        if scenario == 'eq':
            continue
        i += 1

        y_alternatives_scores = row[3:]
        # y_alternatives_ranks = pd.DataFrame(rankdata(y_alternatives_scores, axis=0, method='min'),
        #                                 index=y_alternatives_scores.index)

        plt.subplot(10, 3, i)
        plt.gca().invert_yaxis()
        plt.title(scenario)
        plt.plot(x_values, y_scores_eq, '-')
        plt.plot(x_values, y_alternatives_scores, 'o', label=scenario)

    plt.legend()
    plt.savefig(f'var/weights_scores.png')

if (IS_RUN_CLUSTERS):
    weights_scenarios = {
        'eq': np.full(data.shape[1], 1 / data.shape[1]),
        'crit': critic_weighting(data.to_numpy()),
        'ent': entropy_weighting(data.to_numpy()),
        'gini': gini_weighting(data.to_numpy()),
        'ahp_clusters': np.array(
            [0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062,
             0.0530, 0.0530, 0.0530,
             0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525,
             0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287,
             0.0323, 0.0323, 0.0323])
    }


    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Computed weights of criteria for varied scenarios')
    i = 0
    for scenario, weights in weights_scenarios.items():
        i += 1
        plt.subplot(len(weights_scenarios), 1, i)
        plt.gca().set_ylim([0, 0.6])
        plt.title(scenario)
        plt.plot(data.columns, weights, label=scenario)
        plt.grid(color='whitesmoke', linestyle='solid')
    # plt.legend()
    plt.savefig('var/weights_with_ahp.png')
    # plt.show()
    plt.clf()

    weights_scenarios_pd = pd.DataFrame(weights_scenarios)
    weights_scenarios_pd.to_csv('var/weights_scenarios_with_ahp.csv')



    comparison_ahp = Comparison(data.shape[0], data.shape[1])
    comparison_ahp.add_evaluator('v=0.5', VikorEvaluator(0.5))
    comparison_ahp.add_decision_problem(DECISION_PROBLEM, data, impacts)
    for scenario, weights in weights_scenarios.items():
        comparison_ahp.add_weights_set(scenario, weights)

    comparison_ahp.compute(compute_correlations=True)
    # plt = comparison_ahp.plot_correlations_heatmap(figure_size=(10, 10))
    # plt.savefig(f'var/correlation_with_ahp.png')

    comparison_ahp_df = comparison_ahp.to_dataframe(normalize_scores=False)

    plt.figure(figsize=(20, 10))
    x_values = data.index.to_list()
    for index, row in comparison_ahp_df.iterrows():
        scenario = row['weights_set']
        y_scores = row[3:]
        y_ranks = rankdata(y_scores, axis=0, method='min')

        plt.subplot(1, 2, 1)
        plt.plot(x_values, y_scores, '-o' if scenario == 'ahp_clusters' else ':', label=scenario, linewidth=2 if scenario == 'ahp_clusters' else 1)

        plt.subplot(1, 2, 2)
        plt.plot(x_values, y_ranks, '-o' if scenario == 'ahp_clusters' else ':',label=scenario, linewidth=2 if scenario == 'ahp_clusters' else 1)

    plt.subplot(1, 2, 1)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.gca().invert_yaxis()
    plt.legend()

    plt.savefig(f'var/results_with_ahp.png')


# y_alternatives_ranks = pd.DataFrame(rankdata(y_alternatives_scores, axis=0, method='min'),
#                                             index=y_alternatives_scores.index)
