import pandas as pd
from database import DatabaseManager
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
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

    def create_training_plots(self):
        """
        creates plots for training data
        :return:
        """

        plots = []
        colors = Category10[10]

        for i, (train_col, top_fits) in enumerate(self.best_fit_ranking.items()):

            best_ideal_col, best_ssq = top_fits[0]
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
                    self.ideal_df['X'], self.ideal_df[ideal_col_3rd],
                    line_width=2, line_color="gray", alpha=0.2,
                    legend_label=f"3rd Best ({ideal_col_3rd}): SSQ {ssq_3rd:,.0f}"
                )

            if len(top_fits) > 1:
                ideal_col_2nd, ssq_2nd = top_fits[1]
                p.line(
                    self.ideal_df['X'], self.ideal_df[ideal_col_2nd],
                    line_width=2, line_color="gray", alpha=0.4,
                    legend_label=f"2nd Best ({ideal_col_2nd}): SSQ {ssq_2nd:,.0f}"
                )

            p.line(
                self.ideal_df['X'],
                self.ideal_df[best_ideal_col],
                line_width=2,
                line_color="black",
                legend_label=f"Ideal ({best_ideal_col}): SSQ {best_ssq:,.2f}"
            )

            #training data

            p.scatter(
                x=self.train_df['X'],
                y=self.train_df[train_col],
                size=3,
                alpha=0.5,
                fill_color=colors[i],
                legend_label=f"Training ({train_col})",
                line_color=None
            )

            p.add_tools(HoverTool(tooltips=[("X", "@x"), ("Y", "@y")]))
            plots.append(p)
        return plots


    def create_test_plot(self, test_file_path: str):
        """
        creates a plot for a single test file
        :param test_file_path:
        :return:
        """

        test_data = pd.read_csv(test_file_path).rename(columns={'x': 'X', 'y': 'Y'})

        mapped = self.test_results_df['X (test func)']

        test_data['status'] = 'Unmapped'

        test_data.loc[test_data['X'].isin(mapped), 'status'] = 'Mapped'

        source = ColumnDataSource(test_data)
        colors = ["#CAB2D6", "#FDBF6F"]

        p = figure(
            title=f"Test Data Results for {test_file_path}",
            x_axis_label="X",
            y_axis_label="Y",
            width=800,
            height=400,
        )

        color_mapper = CategoricalColorMapper(factors=['Unmapped', 'Mapped'], palette=colors)

        p.scatter(
            x='X',
            y='Y',
            source=source,
            color={'field': 'status', 'transform': color_mapper},
            legend_label="Status",
            alpha=0.7,
        )

        p.add_tools(HoverTool(tooltips=[("X", "@X"), ("Y", "@Y")]))
        p.legend.location = "top_left"
        return p

    def create_deviation_histogram(self):
        """
        creats a histogram of 'Delta Y'
        :return:
        """

        deviations = self.test_results_df['Delta Y (test func)']

        hist, edges = np.histogram(deviations, density=True, bins=20)

        hist_df = pd.DataFrame({'counts': hist, 'left': edges[:-1], 'right': edges[1:]})
        hist_df['interval'] = [f"{left:.2f} to {right:.2f}" for left, right in zip(hist_df['left'], hist_df['right'])]

        p = figure(
            title="Distribution of Deviations",
            x_axis_label="Deviation",
            y_axis_label="Frequency",
            width=300,
            height=800,
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

    def generate_and_save_plots(self, test_file_path: str):
        """
        generates plots and save to html file
        :param test_file_path:
        :return:
        """

        print(f"Generating plots for {test_file_path}...")
        training_plots = self.create_training_plots()
        test_data_plot = self.create_test_plot(test_file_path)
        deviation_hist = self.create_deviation_histogram()

        if len(training_plots) < 4:
            print("Error: Could not generate training plots. Analysis might have found no matches.")
            layout = gridplot(
                [
                    [test_data_plot],
                    [deviation_hist]
                ],
                sizing_mode="scale_width"
            )
        else:
            layout = gridplot(
                [
                    [training_plots[0], training_plots[1]],
                    [training_plots[2], training_plots[3]],
                    [test_data_plot],
                    [deviation_hist]
                ],
                sizing_mode="scale_width"
            )

        output_file(filename="visuals.html", title="Project Visualization Results")
        save(layout)

        print("Successfully saved visualizations.")