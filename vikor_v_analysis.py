import numpy as np
import pandas as pd
from comparator.comparison import Comparison
from pyrepo_mcda.weighting_methods import critic_weighting, entropy_weighting, gini_weighting
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from vikor_evaluator import VikorEvaluator


class VikorVAnalysis:
    def __init__(self, data_csv_path: str, decision_problem_name='wfarms'):

        self.DECISION_PROBLEM = decision_problem_name
        self.V_MIN = 0
        self.V_MAX = 1
        self.V_STEP = 0.02
        self.BOOSTED_WEIGHT = 0.1
        self.data = None
        self.impacts = None
        self.data = None
        self.v_weights_scenarios = None
        self.comparisons_v = None
        self.weights_scenarios_weights = None
        self.comparison_w = None
        self.weights_scenarios_ahp = None
        self.comparison_ahp = None

        self.load_data(data_csv_path)

    def load_data(self, data_csv_path):
        self.data = pd.read_csv(data_csv_path, index_col=0)
        self.impacts = self.data.iloc[-1]
        self.data = self.data.iloc[:-1]

    def run_experiment_v(self):
        self.v_weights_scenarios = {
            'eq': np.full(self.data.shape[1], 1 / self.data.shape[1]),
            'crit': critic_weighting(self.data.to_numpy()),
            'ent': entropy_weighting(self.data.to_numpy()),
            'gini': gini_weighting(self.data.to_numpy()),
        }

        v_step_rounding = len(f"{self.V_STEP}") - 2
        self.comparisons_v = {}

        # iterate over the scenarios
        for scenario, weights in self.v_weights_scenarios.items():
            comparison_v = Comparison(self.data.shape[0], self.data.shape[1])
            comparison_v.add_decision_problem(self.DECISION_PROBLEM, self.data, self.impacts)
            comparison_v.add_weights_set(scenario, weights)

            for v in np.arange(self.V_MIN, self.V_MAX + (self.V_STEP / 10), self.V_STEP).round(v_step_rounding):
                evaluator = VikorEvaluator(v)
                comparison_v.add_evaluator(f'v_{v}', evaluator)

            comparison_v.compute()

            self.comparisons_v[scenario] = comparison_v

    def draw_v_weights_plots(self, path='var/weights.png', figsize=(10,10), ylim=[0, 0.5]):
        if self.v_weights_scenarios is None:
            self.run_experiment_v()

        plt.clf()

        plt.figure(figsize=figsize)
        i = 0
        for scenario, weights in self.v_weights_scenarios.items():
            i += 1
            plt.subplot(len(self.v_weights_scenarios), 1, i)
            plt.gca().set_ylim(ylim)
            plt.title(scenario)
            plt.plot(self.data.columns, weights, label=scenario)
            plt.grid(color='whitesmoke', linestyle='solid')
        # plt.legend()
        if (path is not None):
            plt.savefig('var/weights.png')
        # plt.show()

    def csv_v_weights(self):
        if self.v_weights_scenarios is None:
            self.run_experiment_v()

        weights_scenarios_pd = pd.DataFrame(self.v_weights_scenarios)
        weights_scenarios_pd.to_csv('var/weights_scenarios.csv')

    def heatmap_v_correlations(self):
        if self.comparisons_v is None:
            self.run_experiment_v()

        for scenario, comparison_v in self.comparisons_v.items():
            print(scenario)
            hplt = comparison_v.plot_correlations_heatmap(figure_size=(50, 20))
            hplt.savefig(f'var/correlation_{scenario}.png')
            # plt.show()

    def sensitivity_analysis_v(self, path_template='var/sensitivity_{}.png'):
        v_step_rounding = len(f"{self.V_STEP}") - 2
        ZOOM_MAX = 2
        for scenario, comparison_v in self.comparisons_v.items():
            x_v_values = np.arange(self.V_MIN, self.V_MAX + (self.V_STEP / 10), self.V_STEP).round(v_step_rounding)
            y_alternatives_scores = pd.DataFrame(comparison_v.scores[self.DECISION_PROBLEM][scenario])
            y_alternatives_ranks = pd.DataFrame(rankdata(y_alternatives_scores, axis=0, method='min'),
                                                index=y_alternatives_scores.index)

            plt.figure(figsize=(20, 10))
            plt.suptitle(f'Sensitivity analysis for scenario {scenario}')

            plt.subplot(1, 2, 1)
            plt.grid(color='whitesmoke', linestyle='solid')
            plt.title('Scores')
            plt.axvline(x=0.5, color='b', linestyle='--')
            plt.gca().invert_yaxis()
            for i in y_alternatives_scores.index:
                plt.plot(x_v_values, y_alternatives_scores.loc[i], label=i)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.grid(color='whitesmoke', linestyle='solid')
            plt.title('Ranks')
            plt.axvline(x=0.5, color='b', linestyle='--')
            plt.gca().invert_yaxis()
            for i in y_alternatives_ranks.index:
                plt.plot(x_v_values, y_alternatives_ranks.loc[i], label=i)
            plt.legend()

            # plt.subplot(1, 3, 3)
            # plt.grid(color='whitesmoke', linestyle='solid')
            # plt.title('Scores Zoomed')
            # plt.gca().invert_yaxis()
            # for i in y_alternatives_scores.index:
            #     plt.plot(x_v_values[:ZOOM_MAX], y_alternatives_scores.loc[i][:ZOOM_MAX], label=i)
            # plt.legend()

            if (path_template is not None):
                plt.savefig(path_template.format(scenario))
            else:
                plt.show()


    def run_experiment_weights(self):
        self.weights_scenarios_weights = {}
        num_criteria = len(self.data.columns)

        # whatever is left to distribute among the non-boosted criteria
        reduced_weight = (1 - self.BOOSTED_WEIGHT) / (num_criteria - 1)

        i = 0
        self.weights_scenarios_weights['eq'] = np.full(num_criteria, 1 / num_criteria)
        for criterion in self.data.columns:
            self.weights_scenarios_weights[criterion] = np.full(num_criteria, reduced_weight)
            self.weights_scenarios_weights[criterion][i] = self.BOOSTED_WEIGHT
            i += 1

        self.comparison_w = Comparison(self.data.shape[0], self.data.shape[1])

        self.comparison_w.add_decision_problem(self.DECISION_PROBLEM, self.data, self.impacts)

        v = 0.5
        evaluator = VikorEvaluator(v)
        self.comparison_w.add_evaluator(f'v_{v}', evaluator)

        for scenario, weights in self.weights_scenarios_weights.items():
            self.comparison_w.add_weights_set(scenario, weights)

        self.comparison_w.compute()

    def draw_weights_sensitivity_plots(self, path='var/weights_sensitivity.png'):
        if self.weights_scenarios_weights is None:
            self.run_experiment_weights()

        plt.figure(figsize=(15, 30))
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(f'Computed weights of criteria for weights sensitivity scenarios')
        i = 0
        for scenario, weights in self.weights_scenarios_weights.items():
            i += 1
            plt.subplot(len(self.weights_scenarios_weights), 1, i)
            plt.gca().set_ylim([0, 0.1])
            plt.title(scenario)
            plt.plot(self.data.columns, weights, label=scenario)
            plt.grid(color='whitesmoke', linestyle='solid')

        if (path is not None):
            plt.savefig(path)

    def heatmap_weights_correlations(self, path='var/correlation_weights.png'):
        if self.comparison_w is None:
            self.run_experiment_weights()

        hplt = self.comparison_w.plot_correlations_heatmap(figure_size=(50, 20))

        if (path is not None):
            hplt.savefig(path)

    def draw_weights_scores(self):
        if self.comparison_w is None:
            self.run_experiment_weights()

        comparison_w_df = self.comparison_w.to_dataframe(normalize_scores=False)

        plt.clf()
        plt.figure(figsize=(10, 20))
        plt.subplots_adjust(hspace=0.5)
        x_values = self.data.index.to_list()
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

    def run_experiment_ahp(self):
        self.weights_scenarios_ahp = {
            'eq': np.full(self.data.shape[1], 1 / self.data.shape[1]),
            'crit': critic_weighting(self.data.to_numpy()),
            'ent': entropy_weighting(self.data.to_numpy()),
            'gini': gini_weighting(self.data.to_numpy()),
            'ahp_clusters': np.array(
                [0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062,
                 0.0530, 0.0530, 0.0530,
                 0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525, 0.0525,
                 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0287,
                 0.0323, 0.0323, 0.0323])
        }

        self.comparison_ahp = Comparison(self.data.shape[0], self.data.shape[1])
        self.comparison_ahp.add_evaluator('v=0.5', VikorEvaluator(0.5))
        self.comparison_ahp.add_decision_problem(self.DECISION_PROBLEM, self.data, self.impacts)
        for scenario, weights in self.weights_scenarios_ahp.items():
            self.comparison_ahp.add_weights_set(scenario, weights)

        self.comparison_ahp.compute(compute_correlations=True)

    def draw_ahp_weights_plots(self):
        if self.weights_scenarios_ahp is None:
            self.run_experiment_ahp()

        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(f'Computed weights of criteria for varied scenarios')
        i = 0
        for scenario, weights in self.weights_scenarios_ahp.items():
            i += 1
            plt.subplot(len(self.weights_scenarios_ahp), 1, i)
            plt.gca().set_ylim([0, 0.6])
            plt.title(scenario)
            plt.plot(self.data.columns, weights, label=scenario)
            plt.grid(color='whitesmoke', linestyle='solid')
        # plt.legend()
        plt.savefig('var/weights_with_ahp.png')
        # plt.show()


    def csv_ahp_weights(self):
        if self.weights_scenarios_ahp is None:
            self.run_experiment_ahp()

        weights_scenarios_df = pd.DataFrame(self.weights_scenarios_ahp)
        weights_scenarios_df.to_csv('var/weights_scenarios_with_ahp.csv')

    def heatmap_ahp_correlations(self):
        if self.comparison_ahp is None:
            self.run_experiment_ahp()

        hplt = self.comparison_ahp.plot_correlations_heatmap(figure_size=(10, 10))
        hplt.savefig(f'var/correlation_with_ahp.png')


    def draw_ahp_results(self):
        if self.comparison_ahp is None:
            self.run_experiment_ahp()

        comparison_ahp_df = self.comparison_ahp.to_dataframe(normalize_scores=False)

        plt.figure(figsize=(20, 10))
        x_values = self.data.index.to_list()
        for index, row in comparison_ahp_df.iterrows():
            scenario = row['weights_set']
            y_scores = row[3:]
            y_ranks = rankdata(y_scores, axis=0, method='min')

            plt.subplot(1, 2, 1)
            plt.plot(x_values, y_scores, '-o' if scenario == 'ahp_clusters' else ':', label=scenario,
                     linewidth=2 if scenario == 'ahp_clusters' else 1)

            plt.subplot(1, 2, 2)
            plt.plot(x_values, y_ranks, '-o' if scenario == 'ahp_clusters' else ':', label=scenario,
                     linewidth=2 if scenario == 'ahp_clusters' else 1)

        plt.subplot(1, 2, 1)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.gca().invert_yaxis()
        plt.legend()

        plt.savefig(f'var/results_with_ahp.png')


if __name__ == '__main__':
    experiment = VikorVAnalysis('wind_farms_data.csv')
    experiment.run_experiment_v()
    experiment.run_experiment_weights()
    experiment.run_experiment_ahp()

    experiment.draw_v_weights_plots()
    experiment.csv_v_weights()
    # experiment.heatmap_v_correlations() #todo sprawdzic dlaczego w EQ jest bialo
    experiment.sensitivity_analysis_v()

    experiment.draw_weights_sensitivity_plots()
    # experiment.heatmap_weights_correlations()
    experiment.draw_weights_scores()

    experiment.draw_ahp_weights_plots()
    experiment.csv_ahp_weights()
    # experiment.heatmap_ahp_correlations()
    experiment.draw_ahp_results()
