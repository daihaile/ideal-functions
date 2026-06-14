import pandas as pd
from database import DatabaseManager, COL_X_TEST, COL_Y_TEST, COL_DELTA_Y_TEST, COL_NO_IDEAL_FUNC, COL_Y_IDEAL, COL_MAPPING_THRESHOLD, COL_ORIGINAL_TRAIN_FUNC
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.io import output_file, save
import numpy as np

class Visualizer:
    """Handles Bokeh visualizations."""

    def __init__(self, db_manager: DatabaseManager, best_fit_ranking: dict):
        """
        Initalizes the visualizer
        :param db_manager:
        :param best_fit_map:
        """
        self.db_manager = db_manager
        self.best_fit_ranking = best_fit_ranking

        #load data from db
        self.train_df = self.db_manager.read_table_to_dataframe("training_data")
        self.ideal_df = self.db_manager.read_table_to_dataframe("ideal_functions")
        self.test_results_df = self.db_manager.read_table_to_dataframe("test_results")

        self.winner_funcs = [v[0][0] for v in self.best_fit_ranking.values()]
        self.colors = Category10[4]
        self.color_map = {func: self.colors[i] for i, func in enumerate(self.winner_funcs)}

    def create_training_plots(self):
        """
        creates plots for training data
        :return:
        """

        plots = []
        colors = Category10[10]

        for i, (train_col, top_fits) in enumerate(self.best_fit_ranking.items()):

            best_ideal_col, best_ssq = top_fits[0]

            plot_data_dict = {
                'X': self.train_df['X'],
                'Y_train': self.train_df[train_col],
                'Y_ideal_1': self.ideal_df[best_ideal_col]
            }

            plot_data_dict['Deviation'] = (plot_data_dict['Y_train'] - plot_data_dict['Y_ideal_1']).abs()

            if len(top_fits) > 1:
                ideal_col_2nd, ssq_2nd = top_fits[1]
                plot_data_dict['Y_ideal_2'] = self.ideal_df[ideal_col_2nd]

            if len(top_fits) > 2:
                ideal_col_3rd, ssq_3rd = top_fits[2]
                plot_data_dict['Y_ideal_3'] = self.ideal_df[ideal_col_3rd]

            source = ColumnDataSource(pd.DataFrame(plot_data_dict))

            p = figure(
                title=f"Training {train_col} vs. Ideal Function for {best_ideal_col}",
                x_axis_label="X",
                y_axis_label="Y",
                width=400,
                height=300,
            )


            if len(top_fits) > 2:
                ideal_col_3rd, ssq_3rd = top_fits[2]
                p.line(
                    x='X', y='Y_ideal_3', source=source,
                    line_width=2, line_color="gray", alpha=0.2,
                    legend_label=f"3rd Best ({ideal_col_3rd}): SSQ {ssq_3rd:,.0f}"
                )

            if len(top_fits) > 1:
                ideal_col_2nd, ssq_2nd = top_fits[1]
                p.line(
                    x='X', y='Y_ideal_2', source=source,
                    line_width=2, line_color="gray", alpha=0.4,
                    legend_label=f"2nd Best ({ideal_col_2nd}): SSQ {ssq_2nd:,.0f}"
                )

            p.line(
                x='X', y='Y_ideal_1',
                source=source,
                line_width=2,
                line_color="black",
                legend_label=f"Ideal ({best_ideal_col}): SSQ {best_ssq:,.2f}"
            )

            #training data

            p.scatter(
                x='X', y='Y_train', source=source,
                size=3,
                alpha=0.5,
                fill_color=colors[i],
                legend_label=f"Training ({train_col})",
                line_color=None
            )

            hover = HoverTool(tooltips=[
                ("X", "@X{0.00}"),
                ("Y (Training)", "@Y_train{0.000}"),
                ("Y (Ideal)", "@Y_ideal_1{0.000}"),
                ("Deviation", "@Deviation{0.000}")
            ])
            p.add_tools(hover)
            p.legend.location = "top_left"
            plots.append(p)
        return plots

    def _prepare_test_data(self, test_file_path: str) -> ColumnDataSource:
        """Loads, merges, and prepares all test data for plotting."""
        all_test_data = pd.read_csv(test_file_path).rename(columns={'x': 'X', 'y': 'Y'})

        results_to_merge = self.test_results_df.rename(columns={
            COL_X_TEST: 'X',
            COL_Y_TEST: 'Y'
        })

        # fix the hover bug by merging X and Y
        all_test_data = pd.merge(all_test_data, results_to_merge, on=['X', 'Y'], how='left')

        # 'status' column will contain ideal function name or 'Unmapped'
        all_test_data['status'] = all_test_data[COL_NO_IDEAL_FUNC].fillna('Unmapped')

        base_size = 4
        max_size = 14

        deviation_col = COL_DELTA_Y_TEST
        # compute sizes if deviation column exists
        if deviation_col in all_test_data.columns:

            # mapped points have assigned ideal function, otherwise unmapped
            mapped_mask = all_test_data['status'] != 'Unmapped'

            # only select mapped rows
            deviations = all_test_data.loc[mapped_mask, deviation_col].astype(float)

            #if mapped points, compute size based on deviation
            if not deviations.empty:
                d_min = float(deviations.min())
                d_max = float(deviations.max())

                if d_max > d_min:
                    # scale deviation into 0 to 1 using min max
                    scaled = (deviations - d_min) / (d_max - d_min)
                    # map marker to sizes
                    all_test_data.loc[mapped_mask, 'plot_size'] = base_size + scaled * (max_size - base_size)
                else:
                    # if d_max == d_min, set size to middle value
                    all_test_data.loc[mapped_mask, 'plot_size'] = (base_size + max_size) / 2

            # unmapped points are assigned size of 4
            all_test_data.loc[~mapped_mask, 'plot_size'] = base_size
        else:
            all_test_data['plot_size'] = base_size

        return ColumnDataSource(all_test_data)

    def _create_color_mapper(self):
        """Creates the color mapper for the test plot."""
        winner_funcs = [v[0][0] for v in self.best_fit_ranking.values()]

        colors = Category10[4]
        color_map = {func: colors[i] for i, func in enumerate(winner_funcs)}

        factors = winner_funcs + ['Unmapped']

        palette = list(colors) + ["#CAB2D6"]

        mapper = CategoricalColorMapper(factors=factors, palette=palette)

        return mapper, winner_funcs, color_map

    def _plot_ideal_lines(self, p: figure, winner_funcs: list, color_map: dict):
        """Plots the 4 ideal function lines on the figure."""
        for func_name in winner_funcs:
            color = color_map[func_name]
            p.line(
                x=self.ideal_df['X'],
                y=self.ideal_df[func_name],
                line_color=color,
                alpha=0.3,
                line_width=3,
                legend_label=f"Ideal: {func_name}"
            )

    def _create_test_hover_tool(self) -> HoverTool:
        """Creates the detailed hover tool for the test plot."""
        return HoverTool(tooltips=[
            ("X", "@X{0.00}"),
            ("Y (Test)", "@Y{0.000}"),
            ("Status", "@status"),
            ("Mapped to", f"@{{{COL_NO_IDEAL_FUNC}}}"),
            ("Orig. Train Func", f"@{{{COL_ORIGINAL_TRAIN_FUNC}}}"),
            ("Y (Ideal)", f"@{COL_Y_IDEAL}{{0.000}}"),
            ("Deviation", f"@{{{COL_DELTA_Y_TEST}}}{{0.0000}}"),
            ("Threshold", f"@{{{COL_MAPPING_THRESHOLD}}}{{0.0000}}")
        ])

    def _create_mapped_plots(self) -> list:
        """
        Creates 4 separate plots, one for each set of mapped test points.
        """
        plots = []

        results_df = self.test_results_df.rename(columns={
            COL_X_TEST: 'X',
            COL_Y_TEST: 'Y',
            COL_NO_IDEAL_FUNC: 'status'
        })

        for func_name, color in self.color_map.items():
            mapped_data = results_df[results_df['status'] == func_name]

            if mapped_data.empty:
                print(f"Warning: No test points were mapped to {func_name}. Skipping plot.")
                plots.append(figure(title=f"No points mapped to {func_name}",
                                    width=400, height=300))
                continue

            source = ColumnDataSource(mapped_data)

            p = figure(
                title=f"Mapped Test Points for {func_name}",
                x_axis_label="X", y_axis_label="Y",
                width=400, height=300
            )

            p.line(
                x=self.ideal_df['X'],
                y=self.ideal_df[func_name],
                line_color=color,
                alpha=0.3,
                line_width=3,
                legend_label=f"Ideal: {func_name}"
            )

            p.scatter(
                x='X',
                y='Y',
                source=source,
                color=color,
                legend_label="Mapped Points",
                alpha=0.7,
                size=5
            )

            hover = HoverTool(tooltips=[
                ("X", "@X{0.00}"),
                ("Y (Test)", "@Y{0.000}"),
                ("Mapped to", "@status"),
                ("Orig. Train Func", f"@{{{COL_ORIGINAL_TRAIN_FUNC}}}"),
                ("Y (Ideal)", f"@{COL_Y_IDEAL}{{0.000}}"),
                ("Deviation", f"@{{{COL_DELTA_Y_TEST}}}{{0.0000}}"),
                ("Threshold", f"@{{{COL_MAPPING_THRESHOLD}}}{{0.0000}}")
            ])
            p.add_tools(hover)
            p.legend.location = "top_left"
            plots.append(p)

        return plots
    def create_deviation_histogram(self):
        """
        creats a histogram of 'Delta Y'
        :return:
        """

        deviations = self.test_results_df[COL_DELTA_Y_TEST]

        hist, edges = np.histogram(deviations, density=True, bins=20)

        hist_df = pd.DataFrame({'counts': hist, 'left': edges[:-1], 'right': edges[1:]})
        hist_df['interval'] = [f"{left:.2f} to {right:.2f}" for left, right in zip(hist_df['left'], hist_df['right'])]

        p = figure(
            title="Distribution of Deviations",
            x_axis_label="Deviation",
            y_axis_label="Frequency",
            width=300,
            height=400,
        )

        p.quad(
            top='counts',
            bottom=0,
            left='left',
            right='right',
            source=hist_df,
            fill_color="#6BAED6",
            line_color="white",
            alpha=0.7
        )

        hover = HoverTool(tooltips=[('Interval', '@interval'), ('Count', '@counts')])
        p.add_tools(hover)
        return p

    def create_test_overview_plot(self, test_file_path: str):
        """Create one overview plot containing all test points, showing mapped and unmapped points"""
        source = self._prepare_test_data(test_file_path)
        mapper, winner_funcs, color_map = self._create_color_mapper()

        p = figure(
            title="All Test Points (Mapped + Unmapped) with size based on deviation",
            x_axis_label="X",
            y_axis_label="Y",
            width=800,
            height=400,
        )

        p.scatter(
            x='X',
            y='Y',
            source=source,
            size='plot_size',
            alpha=0.8,
            color={'field': 'status', 'transform': mapper},
            legend_field='status',
            line_color=None
        )

        hover = self._create_test_hover_tool()
        p.add_tools(hover)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        return p

    def generate_and_save_plots(self, test_file_path: str):
        """
        generates plots and save to html file
        :param test_file_path:
        :return:
        """

        print(f"Generating plots ...")

        test_overview = self.create_test_overview_plot(test_file_path)
        training_plots = self.create_training_plots()
        mapped_plots = self._create_mapped_plots()
        deviation_hist = self.create_deviation_histogram()

        if len(training_plots) < 4 or len(mapped_plots) < 4:
            print("Error: Could not generate all plots. Check analysis results.")
            layout = gridplot(
                [
                    [test_overview],
                    [deviation_hist]
                ],
                sizing_mode="scale_width"
            )
        else:
            layout = gridplot(
                [
                    [test_overview],
                    [training_plots[0], mapped_plots[0]],
                    [training_plots[1], mapped_plots[1]],
                    [training_plots[2], mapped_plots[2]],
                    [training_plots[3], mapped_plots[3]],
                    [deviation_hist]
                ],
                sizing_mode="scale_width"
            )

        output_file(filename="visuals.html", title="Project Visualization Results")
        save(layout)

        print("Successfully saved visualizations.")