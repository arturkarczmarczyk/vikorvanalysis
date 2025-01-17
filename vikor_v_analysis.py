import numpy as np
import pandas as pd
import seaborn as sns
from comparator.comparison import Comparison
from matplotlib import pyplot as plt
from pyrepo_mcda.weighting_methods import critic_weighting
from scipy.stats import rankdata, alpha
from comparator.utils import plot_correlations_heatmap as static_plot_correlations_heatmap

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
        self.comparison_w: Comparison|None = None
        self.weights_scenarios_ahp = None
        self.comparison_ahp: Comparison|None = None

        self.load_data(data_csv_path)

    def load_data(self, data_csv_path):
        self.data = pd.read_csv(data_csv_path, index_col=0)
        self.impacts = self.data.iloc[-1]
        self.data = self.data.iloc[:-1]

    def run_experiment_v(self):
        self.v_weights_scenarios = {
            'eq': np.full(self.data.shape[1], 1 / self.data.shape[1]),
            'crit': critic_weighting(self.data.to_numpy()),
            # 'ent': entropy_weighting(self.data.to_numpy()),
            # 'gini': gini_weighting(self.data.to_numpy()),
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

    def draw_v_weights_plots(self, path='var/weights.png', figsize=(10,10), ylim=[0, 0.5], labels=None):
        if self.v_weights_scenarios is None:
            self.run_experiment_v()

        plt.clf()

        plt.figure(figsize=figsize)
        i = 0
        for scenario, weights in self.v_weights_scenarios.items():
            i += 1
            label = labels[scenario] if labels and scenario in labels else scenario
            plt.subplot(len(self.v_weights_scenarios), 1, i)
            plt.gca().set_ylim(ylim)
            plt.title(label)
            plt.plot(self.data.columns, weights, label=label)
            plt.grid(color='whitesmoke', linestyle='solid')
        # plt.legend()
        if (path is not None):
            plt.savefig(path)
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

        correlations = self.comparison_w.correlations
        new_index = [name.split('-')[1] for name in self.comparison_w.correlations.index]
        new_columns = [name.split('-')[1] for name in self.comparison_w.correlations.columns]
        correlations.index = new_index
        correlations.columns = new_columns

        hplt = static_plot_correlations_heatmap(
            correlations,
            title="Correlation Matrix",
            font_scale=0.7,
            labels_format=".2f",
            color_map="YlGnBu",
            x_label="Boosted weight of criterion",
            y_label="Boosted weight of criterion",
        )
        # hplt = self.comparison_w.plot_correlations_heatmap(
        #     figure_size=(20, 10),
        #     labels_format=".2f",
        #     x_label="Boosted weight of criterion",
        #     y_label="Boosted weight of criterion"
        # )

        if (path is not None):
            hplt.savefig(path)

    def draw_weights_scores(self):
        if self.comparison_w is None:
            self.run_experiment_weights()

        comparison_w_df = self.comparison_w.to_dataframe(normalize_scores=False)

        plt.clf()
        plt.figure(figsize=(12, 10))
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

            plt.subplot(6, 5, i)
            plt.gca().invert_yaxis()
            plt.title(scenario)
            plt.plot(x_values, y_scores_eq, '-')
            plt.plot(x_values, y_alternatives_scores, 'o', label=scenario)

        plt.legend()
        plt.savefig(f'var/weights_scores.pdf')

    def run_experiment_ahp(self):
        self.weights_scenarios_ahp = {
            'eq': np.full(self.data.shape[1], 1 / self.data.shape[1]),
            'crit': critic_weighting(self.data.to_numpy()),
            # 'ent': entropy_weighting(self.data.to_numpy()),
            # 'gini': gini_weighting(self.data.to_numpy()),
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

        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(f'Computed weights of criteria for varied scenarios')
        i = 0
        for scenario, weights in self.weights_scenarios_ahp.items():
            i += 1
            plt.subplot(len(self.weights_scenarios_ahp), 1, i)
            plt.gca().set_ylim([0, 0.125])
            plt.title(scenario)
            plt.plot(self.data.columns, weights, label=scenario)
            plt.grid(color='whitesmoke', linestyle='solid')
        # plt.legend()
        plt.savefig('var/weights_with_ahp.pdf')
        # plt.show()


    def csv_ahp_weights(self):
        if self.weights_scenarios_ahp is None:
            self.run_experiment_ahp()

        weights_scenarios_df = pd.DataFrame(self.weights_scenarios_ahp)
        weights_scenarios_df.to_csv('var/weights_scenarios_with_ahp.csv')

    def heatmap_ahp_correlations(self):
        if self.comparison_ahp is None:
            self.run_experiment_ahp()

        correlations = self.comparison_ahp.correlations
        new_names = ['Equal weights', 'CRITIC weights', 'AHP clusters']
        correlations.index = new_names
        correlations.columns = new_names

        hplt = static_plot_correlations_heatmap(
            correlations,
            title="",
            font_scale=0.8,
            labels_format=".2f",
            color_map="Blues",
            x_label="",
            y_label="",
            figure_size=(5,3)
        )

        # hplt = self.comparison_ahp.plot_correlations_heatmap(figure_size=(10, 10))
        hplt.savefig(f'var/correlation_with_ahp.pdf')


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

        plt.savefig(f'var/results_with_ahp.pdf')

    def run_experiment_criteria_elimination(self, limited_data, limited_impacts, ahp_weights):
        if (self.weights_scenarios_ahp is None):
            self.run_experiment_ahp()

        self.limited_data = limited_data
        self.limited_impacts = limited_impacts

        self.limited_weights_scenarios_ahp = {
            'eq': np.full(self.limited_data.shape[1], 1 / self.limited_data.shape[1]),
            'crit': critic_weighting(self.limited_data.to_numpy()),
            # 'ent': entropy_weighting(self.limited_data.to_numpy()),
            # 'gini': gini_weighting(self.limited_data.to_numpy()),
            'ahp_clusters': ahp_weights
        }

        self.comparison_criteria_elimination = Comparison(self.limited_data.shape[0], self.limited_data.shape[1])
        self.comparison_criteria_elimination.add_evaluator('v=0.5', VikorEvaluator(0.5))
        self.comparison_criteria_elimination.add_decision_problem(self.DECISION_PROBLEM, self.limited_data, self.limited_impacts)
        for scenario, weights in self.limited_weights_scenarios_ahp.items():
            self.comparison_criteria_elimination.add_weights_set(scenario, weights)

        self.comparison_criteria_elimination.compute(compute_correlations=True)

    def draw_elimination_results_comparison(self, scenarios: list[str], path = None):
        if self.comparison_criteria_elimination is None:
            raise ValueError("First run the experiment")

        comparison_ahp_df = self.comparison_ahp.to_dataframe(normalize_scores=False)
        comparison_elim_df = self.comparison_criteria_elimination.to_dataframe(normalize_scores=False)

        plt.clf()

        num_rows = len(scenarios)
        num_cols = 2
        plt.figure(figsize=(num_cols * 5, num_rows * 5))
        x_values = self.data.index.to_list()

        current_row = 0
        for scenario in scenarios:
            current_row += 1

            scenario_ahp_row = comparison_ahp_df[comparison_ahp_df['weights_set'] == scenario].iloc[0]
            y_scores_ahp = scenario_ahp_row[3:]
            y_ranks_ahp = pd.DataFrame(rankdata(y_scores_ahp, axis=0, method='min'), index=y_scores_ahp.index)

            scenario_elim_row = comparison_elim_df[comparison_elim_df['weights_set'] == scenario].iloc[0]
            y_scores_elim = scenario_elim_row[3:]
            y_ranks_elim = pd.DataFrame(rankdata(y_scores_elim, axis=0, method='min'), index=y_scores_elim.index)

            plt.subplot(num_rows,num_cols, ((current_row - 1) * num_cols) + 1)
            plt.title(f"{scenario} scores")
            plt.plot(x_values, y_scores_ahp, label=f"baseline")
            plt.plot(x_values, y_scores_elim, label=f"with eliminated criteria")
            plt.gca().invert_yaxis()
            plt.legend()

            # plt.subplot(num_rows,num_cols, ((current_row - 1) * num_cols) + 2)
            # plt.title("Ranks")
            # plt.plot(x_values, y_ranks_ahp, label=f"Baseline {scenario}")
            # plt.plot(x_values, y_ranks_elim, label=f"{scenario} with eliminated criteria")
            # plt.gca().invert_yaxis()
            # plt.legend()

            plt.subplot(num_rows,num_cols, ((current_row - 1) * num_cols) + 2)
            plt.title(f"{scenario} ranks")
            num_alternatives = len(x_values)
            ranks = np.arange(1, num_alternatives + 1)
            plt.plot(ranks, ranks, '-', color="black")
            plt.plot(y_ranks_ahp, y_ranks_elim, 'o')
            # Annotate each point
            for i, (ahp_rank, elim_rank) in enumerate(zip(y_ranks_ahp.values, y_ranks_elim.values)):
                plt.annotate(f'A{i + 1}', (ahp_rank, elim_rank), textcoords="offset points", xytext=(0, 10),
                             ha='center')

            plt.xlabel(f"baseline ranks")
            plt.ylabel(f"ranks with eliminated criteria")

            plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        if (path is not None):
            plt.savefig(path)

    def heatmap_elimination_correlations(self, path=None):
        if self.comparison_criteria_elimination is None:
            raise ValueError("First run the experiment")

        correlations = self.comparison_criteria_elimination.correlations
        new_names = ['Equal weights', 'CRITIC weights', 'AHP clusters']
        correlations.index = new_names
        correlations.columns = new_names

        hplt = static_plot_correlations_heatmap(
            correlations,
            title="",
            font_scale=0.8,
            labels_format=".2f",
            color_map="Blues",
            x_label="",
            y_label="",
            figure_size=(5, 3)
        )

        # hplt = self.comparison_criteria_elimination.plot_correlations_heatmap(figure_size=(10, 10))

        if (path is not None):
            hplt.savefig(path)

    def draw_ahp_and_elim_weights_plots(self):
        if self.weights_scenarios_ahp is None or self.limited_weights_scenarios_ahp is None:
            raise ValueError("First run the experiment")

        plt.figure(figsize=(10, 8))
        # plt.suptitle(f'Computed weights of criteria for varied scenarios')
        i = 0
        for scenario, weights in self.weights_scenarios_ahp.items():
            i += 1
            plt.subplot(len(self.weights_scenarios_ahp), 1, i)
            plt.title(scenario)
            plt.plot(self.data.columns, weights, label=f"{scenario} - baseline")
            plt.plot(self.limited_data.columns, self.limited_weights_scenarios_ahp[scenario], 'x', label=f"{scenario} - eliminated criteria")
            plt.grid(color='whitesmoke', linestyle='solid')
            plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('var/weights_with_ahp_eliminated.pdf')
        # plt.show()

    def run_experiment_financial_efficiency(self, criteria_for_denominator):
        self.numerator_data = self.data.drop(columns=criteria_for_denominator)
        self.numerator_impacts = self.impacts.drop(index=criteria_for_denominator)

        self.denominator_data = self.data[criteria_for_denominator]
        self.denominator_values = self.denominator_data.sum(axis=1)

        self.financial_efficiency_weights_scenarios = {
            'eq': np.full(self.numerator_data.shape[1], 1 / self.numerator_data.shape[1]),
            'crit': critic_weighting(self.numerator_data.to_numpy()),
        }

        self.comparison_financial_efficiency = Comparison(self.numerator_data.shape[0], self.numerator_data.shape[1])
        self.comparison_financial_efficiency.add_evaluator('v=0.5', VikorEvaluator(0.5))
        self.comparison_financial_efficiency.add_decision_problem(self.DECISION_PROBLEM, self.numerator_data, self.numerator_impacts)
        for scenario, weights in self.financial_efficiency_weights_scenarios.items():
            self.comparison_financial_efficiency.add_weights_set(scenario, weights)

        self.comparison_financial_efficiency.compute(compute_correlations=True)
        scores_fe = self.comparison_financial_efficiency.to_dataframe()
        scores_fe[scores_fe.columns[3:]] = (1 / scores_fe[scores_fe.columns[3:]]) / self.denominator_values
        self.scores_financial_efficiency = scores_fe

    def dataframe_financial_efficiency_scores(self):
        if self.scores_financial_efficiency is None:
            raise ValueError("First run the experiment")

        concated = pd.concat([
            self.comparison_financial_efficiency.to_dataframe().assign(scenario='baseline'),
            self.scores_financial_efficiency.assign(scenario='financial_efficiency')
        ])
        concated = concated[['decision_problem', 'weights_set', 'evaluator', 'scenario', *self.data.index]]

        return concated

    def draw_financial_efficiency_ranks(self, path=None):
        if self.scores_financial_efficiency is None:
            raise ValueError("First run the experiment")

        alternatives = self.numerator_data.index.to_list()

        plt.clf()
        plt.figure(figsize=(10,5))

        for index, row in self.dataframe_financial_efficiency_scores().iterrows():
            subplot = 1 if row['weights_set'] == 'eq' else 2
            order = 1 if row['scenario'] == 'baseline' else -1

            plt.subplot(1, 2, subplot)
            scores = row[4:]
            ranks = pd.DataFrame(rankdata([order * i for i in scores], axis=0, method='min'),
                                 index=scores.index)
            label = f"{row['scenario']} - {row['weights_set']}"

            plt.plot(alternatives, ranks, label=label)
            plt.legend()

        plt.subplot(1,2,1)
        plt.gca().invert_yaxis()
        plt.subplot(1,2,2)
        plt.gca().invert_yaxis()

        if (path is not None):
            plt.savefig(path)

    def get_all_ranks(self):
        df_comparisons_v_eq = self.comparisons_v['eq'].to_dataframe()
        # df_comparisons_v_eq = df_comparisons_v_eq[df_comparisons_v_eq['evaluator'].isin(['v_0.1', 'v_0.5', 'v_1.0'])]

        df_comparisons_v_crit = self.comparisons_v['crit'].to_dataframe()
        # df_comparisons_v_crit = df_comparisons_v_crit[df_comparisons_v_crit['evaluator'].isin(['v_0.0', 'v_0.5', 'v_1.0'])]

        all_scores = pd.concat([
            df_comparisons_v_eq,
            df_comparisons_v_crit,
            self.comparison_w.to_dataframe(),
            self.comparison_ahp.to_dataframe(),
            self.comparison_criteria_elimination.to_dataframe(),
        ])
        only_scores = all_scores[all_scores.columns[3:]]
        ranks = only_scores.rank(axis=1, method='min', ascending=True)

        # eliminate rows in which sum of all cells is 7 (ie all ranks are the same)
        ranks = ranks[ranks.sum(axis=1) != 7]

        return ranks

    def get_min_max_average_ranks(self):
        all_ranks = self.get_all_ranks()

        min_ranks = all_ranks.min()
        max_ranks = all_ranks.max()
        avg_ranks = all_ranks.mean()

        ranges = pd.DataFrame({
            'min': min_ranks,
            'max': max_ranks,
            'avg': avg_ranks,
        })

        return ranges

    def draw_rank_intervals(self):
        all_ranks = self.get_all_ranks()
        ranges = self.get_min_max_average_ranks()

        # colors = plt.cm.tab20.colors[:all_ranks.shape[1]]

        x = np.arange(1, all_ranks.shape[1] + 1)

        for (xx, row) in zip(x, ranges.iterrows()):
            plt.plot([xx, xx, xx], [row[1]['min'], row[1]['max'], row[1]['avg']], 'x--', markersize=8)

        for ranks in all_ranks.to_numpy():
            plt.plot(x+0.1, ranks, '.', markersize=10, alpha=0.008, color='black')

        plt.grid(axis='both', alpha=0.7)
        plt.xticks(x, all_ranks.columns, fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Alternatives', fontsize=14)
        plt.ylabel('Rank', fontsize=14)
        plt.title('Maximum, minimum and average rank all alternatives', fontsize=16)
        plt.tight_layout()

        plt.gca().invert_yaxis()

        plt.show()

    def draw_all_ranks_heatmap(self):
        all_ranks = self.get_all_ranks()

        alternative_positions_counts = {}

        # go over all all_ranks columns (alternatives) and for each row (scenario) increase the alternative_positions_counts[alternative][rank] by one, where rank is the value at each cell for each alternative and scenario
        for alternative in all_ranks.columns:
            alternative_positions_counts[alternative] = {}

            # create an index for each possible rank (1 to count(alternatives))
            for rank in range(1, all_ranks.shape[1] + 1):
                alternative_positions_counts[alternative][rank] = 0

            for rank in all_ranks[alternative]:
                alternative_positions_counts[alternative][rank] += 1

        # convert to dataframe
        alternative_positions_counts_df = pd.DataFrame(alternative_positions_counts)

        # convert values to percentages of total scenarios (divide by number of scenarios)
        alternative_positions_counts_df = alternative_positions_counts_df / all_ranks.shape[0]

        plt.clf()
        plt.figure(figsize=(8,6))
        ax = plt.gca()

        matrix = alternative_positions_counts_df.to_numpy()
        sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": "Probability of having the rank"}, ax=ax)
        ax.figure.axes[-1].yaxis.label.set_size(10)
        ax.set_title("", fontsize=12)
        x = np.arange(0, matrix.shape[0])
        ax.set_xticks(x + 0.5, [f'$A_{{{i + 1}}}$' for i in range(len(x))], fontsize=8)
        ax.set_yticks(x + 0.5, [f'${i + 1}$' for i in range(len(x))], fontsize=8)
        ax.set_xlabel("Alternatives", fontsize=10)
        ax.set_ylabel("Ranks", fontsize=10)

        plt.tight_layout()
        plt.savefig('var/fig-generalised-robustness-map.pdf')
        # plt.show()



if __name__ == '__main__':
    experiment = VikorVAnalysis('wind_farms_data.csv')
    experiment.run_experiment_v()
    experiment.run_experiment_weights()
    experiment.run_experiment_ahp()
    experiment.run_experiment_criteria_elimination(
        experiment.data.drop(columns=['S16', 'S17', 'F28', 'F29', 'F30']),
        experiment.impacts.drop(index=['S16', 'S17', 'F28', 'F29', 'F30']),
        ahp_weights=np.array([
            0.041/6, 0.041/6, 0.041/6, 0.041/6, 0.041/6, 0.041/6,
            0.176/3, 0.176/3, 0.176/3,
            0.465/6, 0.465/6, 0.465/6, 0.465/6, 0.465/6, 0.465/6, # updated from 8
            0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10, 0.318/10,
            # financial costs removed completely
        ])
    )
    # experiment.run_experiment_financial_efficiency(['F28', 'F29', 'F30'])
    # experiment.draw_v_weights_plots(path='var/weights.pdf', ylim=[0, 0.12], labels={'eq': 'Equal weights', 'crit': 'CRITIC weights'})
    # experiment.csv_v_weights()
    # # experiment.heatmap_v_correlations() #todo sprawdzic dlaczego w EQ jest bialo
    # experiment.sensitivity_analysis_v(path_template="var/fig-v-sensitivity_{}.pdf")
    #
    # experiment.draw_weights_sensitivity_plots('var/weights_sensitivity.png')
    # experiment.heatmap_weights_correlations('var/fig-correlation-weights.pdf')
    # experiment.draw_weights_scores()
    #
    # experiment.draw_ahp_weights_plots()
    # experiment.csv_ahp_weights()
    # experiment.heatmap_ahp_correlations()
    # experiment.draw_ahp_results()
    #
    # experiment.draw_elimination_results_comparison(['eq', 'crit', 'ahp_clusters'], path='var/results_with_ahp_eliminated.pdf')
    # plt.show()
    # experiment.draw_ahp_and_elim_weights_plots()
    # experiment.heatmap_elimination_correlations('var/fig-correlation-with-elimination.pdf')

    # experiment.draw_financial_efficiency_ranks(path='var/financial_efficiency_ranks.png')

    # ranks = experiment.draw_rank_intervals()
    experiment.draw_all_ranks_heatmap()
