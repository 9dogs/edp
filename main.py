# -*- coding: utf-8 -*-
"""
Echo data calculation software
"""
# Major library imports
import numpy as np
import pandas as pd

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


from lmfit import Model, Parameters
import seaborn as sns
sns.set(style="whitegrid")


LASER_TITLE = 'Laser'
DELAY_TITLE = 'Time'


def echo_decay(x, y0, A, t2):
    return y0 + A*np.exp(-4 * x / t2)


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        self.setWindowTitle("Echo Data Processing")
        self.resize(1100, 600)

        # Массив с данными
        self.data = pd.DataFrame()
        self.data_filename = ""

        self._central_widget = QtGui.QWidget()

        self._graph_tabs_widget = QtGui.QTabWidget()
        self._graph_tabs_widget.currentChanged.connect(self._on_tab_changed)

        self._main_layout = QtGui.QHBoxLayout()
        self._pyqtgraph_layout = QtGui.QVBoxLayout()

        # GraphLayoutWidget с двумя графиками: гистограмма лазера и эхо
        self._plot_area = pg.GraphicsLayoutWidget()
        self.laser_plot = self._plot_area.addPlot()
        self._plot_area.nextRow()
        self.echo_plot = self._plot_area.addPlot()

        # ComboBox для выбора графика
        self._graph_select = QtGui.QComboBox()
        self._graph_select.currentIndexChanged.connect(self._calc_curve_data)

        # Область выделения
        self._select_region = pg.LinearRegionItem()
        self._select_region.setZValue(-10)
        self._select_region.sigRegionChanged.connect(self._calc_curve_data)

        # Область выделения для фита
        self._fit_region = pg.LinearRegionItem()
        self._fit_region.setZValue(-10)
        self._fit_region.sigRegionChanged.connect(self._update_statusbar)

        # Посчитана ли статистика фита
        self._fit_stats = False

        # Результирующие графики

        # График фита
        fitting_tab = QtGui.QWidget()
        # a figure instance to plot on
        self._fitting_figure = plt.figure()
        self._fitting_canvas = FigureCanvas(self._fitting_figure)
        # toolbar
        fitting_toolbar = NavigationToolbar(self._fitting_canvas, self)
        # fit parameters
        params_layout = QtGui.QHBoxLayout()
        # y0
        y0_form = QtGui.QFormLayout()
        self.y0_min, self.y0_max, self.y0_value, self.y0_fixed = QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QCheckBox()
        for spin_box in [self.y0_min, self.y0_max, self.y0_value]:
            spin_box.setMaximum(99999)
            spin_box.setMinimum(-99999)
            spin_box.setSingleStep(0.001)
        y0_form.addRow(QtGui.QLabel("y0 (~5000)"))
        y0_form.addRow("Min", self.y0_min)
        y0_form.addRow("Max", self.y0_max)
        y0_form.addRow("Value", self.y0_value)
        y0_form.addRow("Fixed?", self.y0_fixed)

        params_layout.addLayout(y0_form)

        # A
        A_form = QtGui.QFormLayout()
        self.A_min, self.A_max, self.A_value, self.A_fixed = QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QCheckBox()
        for spin_box in [self.A_min, self.A_max, self.A_value]:
            spin_box.setMaximum(99999)
            spin_box.setMinimum(-99999)
            spin_box.setSingleStep(0.001)
        A_form.addRow(QtGui.QLabel("A (~500)"))
        A_form.addRow("Min", self.A_min)
        A_form.addRow("Max", self.A_max)
        A_form.addRow("Value", self.A_value)
        A_form.addRow("Fixed?", self.A_fixed)

        params_layout.addLayout(A_form)

        # t2
        t2_form = QtGui.QFormLayout()
        self.t2_min, self.t2_max, self.t2_value, self.t2_fixed = QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QDoubleSpinBox(), QtGui.QCheckBox()
        for spin_box in [self.t2_min, self.t2_max, self.t2_value]:
            spin_box.setMaximum(99999)
            spin_box.setMinimum(-99999)
            spin_box.setSingleStep(0.001)
        t2_form.addRow(QtGui.QLabel("t2 (~500)"))
        t2_form.addRow("Min", self.t2_min)
        t2_form.addRow("Max", self.t2_max)
        t2_form.addRow("Value", self.t2_value)
        t2_form.addRow("Fixed?", self.t2_fixed)

        params_layout.addLayout(t2_form)

        # Initial guesses
        self.y0_value.setValue(5000)
        self.A_value.setValue(500)
        self.t2_value.setValue(500)

        fit_layout = QtGui.QVBoxLayout()
        fit_layout.addWidget(fitting_toolbar)
        fit_layout.addWidget(self._fitting_canvas)
        fit_layout.addLayout(params_layout)
        fitting_tab.setLayout(fit_layout)

        self._graph_tabs_widget.addTab(fitting_tab, 'Фит')

        # График зависимости t2 от точки старта фита
        fitting_stats_tab = QtGui.QWidget()
        # a figure instance to plot on
        self._fitting_stats_figure = plt.figure()
        self._fitting_stats_canvas = FigureCanvas(self._fitting_stats_figure)
        fitting_stats_toolbar = NavigationToolbar(self._fitting_stats_canvas, self)

        fit_stats_layout = QtGui.QVBoxLayout()
        fit_stats_layout.addWidget(fitting_stats_toolbar)
        fit_stats_layout.addWidget(self._fitting_stats_canvas)
        fitting_stats_tab.setLayout(fit_stats_layout)

        self._graph_tabs_widget.addTab(fitting_stats_tab, 'Зависимость фита')

        # Фит репорт
        fit_report_widget = QtGui.QWidget()
        report_layout = QtGui.QVBoxLayout()
        self.fit_report_text = QtGui.QTextEdit()
        save_report_btn = QtGui.QPushButton("Сохранить")
        save_report_btn.clicked.connect(self._on_save_report_btn_clicked)
        report_layout.addWidget(self.fit_report_text)
        report_layout.addWidget(save_report_btn)
        fit_report_widget.setLayout(report_layout)
        self._graph_tabs_widget.addTab(fit_report_widget, 'Отчет')

        # Бины для гистограммы лазера
        self._bins_widget = QtGui.QDoubleSpinBox()
        self._bins_widget.setMinimum(1)
        self._bins_widget.setMaximum(1000)
        self._bins_widget.setValue(200)
        self._bins_widget.valueChanged.connect(self._on_bins_value_changed)

        # Вертикальная линия для фита
        self._v_line = pg.InfiniteLine(angle=90, movable=True)
        self._v_line.sigPositionChangeFinished.connect(self._on_v_line_pos_changed)

        # Кпопка копирования в буфер
        self._copy_data_btn = QtGui.QPushButton("Скопировать в буфер")
        self._copy_data_btn.clicked.connect(self._on_copy_data_clicked)

        # Кнопка фита
        self._fit_btn = QtGui.QPushButton("Фитировать")
        self._fit_btn.clicked.connect(self._on_fit_btn_clicked)
        self._fit_btn.setToolTip("Shift+Click to calc t2-start fit dependency")

        # Галочка логарифмического масштаба
        self._log_checkbox = QtGui.QCheckBox("Лог. масштаб")
        self._log_checkbox.stateChanged.connect(self._on_log_scale_changed)

        # Галочка "нормировать на лазер"
        self._div_laser_checkbox = QtGui.QCheckBox("Норм. на лазер")
        self._div_laser_checkbox.stateChanged.connect(self._div_laser_changed)

        # Layout для выбора графиков и бинов
        self._aux_layout = QtGui.QHBoxLayout()
        self._aux_layout.addWidget(self._bins_widget)
        self._aux_layout.addWidget(self._graph_select)
        self._aux_layout.addWidget(self._log_checkbox)
        self._aux_layout.addWidget(self._div_laser_checkbox)
        self._aux_layout.addStretch(5)
        self._aux_layout.addWidget(self._fit_btn)
        self._aux_layout.addWidget(self._copy_data_btn)

        self._pyqtgraph_layout.addWidget(self._plot_area)
        self._pyqtgraph_layout.addLayout(self._aux_layout)

        # Установка лэйаутов
        self._main_layout.addLayout(self._pyqtgraph_layout)
        self._main_layout.addWidget(self._graph_tabs_widget)
        self._central_widget.setLayout(self._main_layout)
        self.setCentralWidget(self._central_widget)

        self._create_actions()
        self._create_menu()
        self._create_status_bar()

    def _on_tab_changed(self, tab):
        pass

    def _div_laser_changed(self, state):
        self._calc_curve_data()

    def _on_save_report_btn_clicked(self):
        report_filename = self.data_filename + ".report"
        with open(report_filename, 'w') as report_file:
            report_file.write("Fit report for: {}".format(self.data_filename))
            report_file.write(self.fit_report_text.toPlainText())

    def _on_log_scale_changed(self, state):
        if state == 0:
            # Log scale unchecked
            self.echo_plot.setLogMode(x=False)
            fit_start, fit_end = [10 ** float(self._fit_region.getRegion()[0]),
                                  10 ** float(self._fit_region.getRegion()[1])]
            self._fit_region.setRegion([fit_start, fit_end])
        elif state == 2:
            # Log scale checked
            self.echo_plot.setLogMode(x=True)
            fit_start, fit_end = self._fit_region.getRegion()
            if fit_start <= 0:
                fit_start = 0.01
            if fit_end <= 0:
                fit_end = 1
            else:
                fit_start = np.log10(fit_start)
                fit_end = np.log10(fit_end)
            self._fit_region.setRegion([fit_start, fit_end])
            self.echo_plot.setXRange(0, 3)

    def _calc_fit_start_dep_curve(self):
        t2_array = []
        t2_var_array = []
        x_array = []
        # Fit value
        fit_variable_name = self._graph_select.currentText()
        for start_fit_from in self.means[DELAY_TITLE]:
            start_fit_from = float(start_fit_from)
            fit_means = self.means[start_fit_from < self.means[DELAY_TITLE]]
            fit_std = self.std[start_fit_from < self.std[DELAY_TITLE]]

            params = self._init_params()

            model = Model(echo_decay, independent_vars=['x'])
            try:
                result = model.fit(fit_means[fit_variable_name], x=fit_means[DELAY_TITLE],
                                   params=params, weights=1 / fit_std[LASER_TITLE] ** 2)
                t2_array.append(result.params.get('t2').value)
                t2_var_array.append(result.params.get('t2').stderr)
                x_array.append(start_fit_from)
            except TypeError:
                break

        ax = self._fitting_stats_figure.add_subplot(111)
        ax.errorbar(x_array, t2_array, yerr=t2_var_array, fmt='--bo', capthick=2, ecolor='b', alpha=0.9)
        ax.set_xlim([-10, 100])
        ax.set_ylim([0, 2000])
        ax.set_title("Зависимость $T_2$ от точки старта фитирования")
        # refresh canvas and tight layout
        self._fitting_stats_canvas.draw()
        self._fitting_stats_figure.tight_layout()
        self._fit_stats = True
        # Adding to fit report
        self.fit_report_text.append("Зависимость t2 от точки старта фитирования:\n")
        self.fit_report_text.append(" ".join("{:.4f}: {:.2f}".format(x, y) for x, y in zip(x_array, t2_array)))
        self.fit_report_text.append("\n")

    def _on_v_line_pos_changed(self):
        self._calc_curve_data()

    def _on_copy_data_clicked(self):
        if not self.means.empty:
            self.means.to_clipboard()

    def _init_params(self):
        # Params
        params = Parameters()
        y0_min = self.y0_min.value() if self.y0_min.value() > 0 else None
        y0_max = self.y0_max.value() if self.y0_max.value() > 0 else None
        y0_value = self.y0_value.value()
        y0_vary = False if self.y0_fixed.isChecked() else True
        params.add('y0', min=y0_min, max=y0_max, value=y0_value, vary=y0_vary)

        A_min = self.A_min.value() if self.A_min.value() > 0 else None
        A_max = self.A_max.value() if self.A_max.value() > 0 else None
        A_value = self.A_value.value()
        A_vary = False if self.A_fixed.isChecked() else True
        params.add('A', min=A_min, max=A_max, value=A_value, vary=A_vary)

        t2_min = self.t2_min.value() if self.t2_min.value() > 0 else None
        t2_max = self.t2_max.value() if self.t2_max.value() > 0 else None
        t2_value = self.t2_value.value()
        t2_vary = False if self.t2_fixed.isChecked() else True
        params.add('t2', min=t2_min, max=t2_max, value=t2_value, vary=t2_vary)

        return params

    def _fit_data(self):
        # clean plot
        self._fitting_figure.clear()

        # Check if x scale is log
        if self._log_checkbox.isChecked():
            fit_start, fit_end = [10 ** float(self._fit_region.getRegion()[0]),
                                  10 ** float(self._fit_region.getRegion()[1])]
        else:
            fit_start, fit_end = [float(self._fit_region.getRegion()[0]), float(self._fit_region.getRegion()[1])]
        fit_means = self.means[(fit_start < self.means[DELAY_TITLE]) & (self.means[DELAY_TITLE] < fit_end)]
        fit_std = self.std[(fit_start < self.std[DELAY_TITLE]) & (self.means[DELAY_TITLE] < fit_end)]

        # Fit value
        fit_variable_name = self._graph_select.currentText()

        params = self._init_params()

        model = Model(echo_decay, independent_vars=['x'])
        result = model.fit(fit_means[fit_variable_name], x=fit_means[DELAY_TITLE], params=params,
                           weights=1 / fit_std[LASER_TITLE] ** 2)

        time = np.linspace(fit_start, fit_end, 1000)
        all_time = np.linspace(self.means[DELAY_TITLE].min(), self.means[DELAY_TITLE].max(), 2000)

        ax = self._fitting_figure.add_subplot(111)
        # plot data
        ax.errorbar(self.means[DELAY_TITLE], self.means[fit_variable_name], yerr=self.std[fit_variable_name],
                    fmt='o', capthick=2, ecolor='b', alpha=0.5)
        ax.plot(time, echo_decay(x=time, **result.values), '-r', alpha=0.9, linewidth=2)
        ax.plot(all_time, echo_decay(x=all_time, **result.values), '--k', alpha=0.4)
        ax.set_ylim([self.means[fit_variable_name].min() - self.std[fit_variable_name].mean(),
                     self.means[fit_variable_name].max() + self.std[fit_variable_name].mean()])

        # Подсчитываем данные
        t2 = result.values['t2']
        t2_stderr = result.params.get('t2').stderr
        # Ширина линии в MHz
        delta_f = 10**6/(np.pi*result.values['t2'])
        delta_f_stderr = delta_f*t2_stderr/t2
        ax.set_title("$T_2$: {:0.2f} $\pm$ {:0.2f} $ps$ ({:0.2f} $\pm$ {:0.2f} $MHz$ )".format(t2, t2_stderr,
                                                                                               delta_f, delta_f_stderr))
        # refresh canvas and tight layout
        self._fitting_canvas.draw()
        self._fitting_figure.tight_layout()
        # Populating report
        self.fit_report_text.append(result.fit_report())
        # Populating spinboxes
        self.y0_value.setValue(result.values['y0'])
        self.A_value.setValue(result.values['A'])
        self.t2_value.setValue(result.values['t2'])

    def _on_fit_btn_clicked(self):
        self._fit_data()
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            self._calc_fit_start_dep_curve()

    def _update_line_pos(self):
        try:
            self._v_line.setPos(self.means.mean())
        except ValueError:
            pass

    def _fill_graph_select(self):
        """
        Функция заполняет ComboBox названиями колонок из данных
        """
        if not self.data.empty:
            for item in self.data.columns:
                self._graph_select.addItem(item)

    def _on_region_changed(self):
        pass

    def _reset_fit_initials(self):
        self.y0_value.setValue(5000)
        self.t2_value.setValue(500)
        self.A_value.setValue(500)

    def _on_bins_value_changed(self):
        """
        Функция срабатывает при изменении бинов
        """
        if not self.data.empty:
            y, x = np.histogram(self.data[LASER_TITLE], bins=self._bins_widget.value())
            curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(255, 255, 255, 40))
            self.laser_plot.clear()
            self.laser_plot.addItem(curve)
            std = self.data[LASER_TITLE].std()
            self._select_region.setRegion([self.data[LASER_TITLE].mean() - std, self.data[LASER_TITLE].mean() + std])
            self.laser_plot.addItem(self._select_region)

    def _create_menu(self):
        self._file_menu = self.menuBar().addMenu("&Файл")
        self._file_menu.addAction(self._open_action)
        self._file_menu.addAction(self._exit_action)

        self._fit_menu = self.menuBar().addMenu("&Фитинг")
        self._fit_menu.addAction(self._fit_action)
        self._fit_menu.addAction(self._reset_initials_action)

        self._help_menu = self.menuBar().addMenu("&Помощь")
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def _calc_curve_data(self):
        """
        Calculates data for echo plot
        """
        self._laser_min, self._laser_max = self._select_region.getRegion()
        # Getting data between laser_min and laser_max
        good_data = self.data[(float(self._laser_min) < self.data[LASER_TITLE]) & (self.data[LASER_TITLE] < float(self._laser_max))]
        self._del_rows_count = (self.data.shape[0] - good_data.shape[0])/self.data.shape[0] * 100
        self._laser_mean = good_data[LASER_TITLE].mean()
        self._laser_std = good_data[LASER_TITLE].std()

        averaged_data = good_data.groupby(DELAY_TITLE)
        self.means = averaged_data.mean().reset_index()
        # Divided by laser
        self.means['normed'] = self.means[self._graph_select.currentText()] / self.means[LASER_TITLE]
        self.std = averaged_data.std().reset_index()
        self.count = averaged_data.count().reset_index()

        self.echo_plot.clear()
        plot_column = 'normed' if self._div_laser_checkbox.isChecked() else self._graph_select.currentText()
        self.echo_plot.plot(self.means[DELAY_TITLE], self.means[plot_column], pen=(200, 200, 200), symbolBrush=(230, 0, 0, 0.8*255), symbolPen='w')
        err = pg.ErrorBarItem(x=self.means[DELAY_TITLE], y=self.means[self._graph_select.currentText()],
                              height=2*self.std[self._graph_select.currentText()], beam=0.5)
        # self.echo_plot.addItem(err)
        # self._fit_region.setRegion([0, 1])
        self.echo_plot.addItem(self._fit_region)
        self._update_statusbar()
        # self._update_line_pos()

    def _update_statusbar(self):
        if self._log_checkbox.isChecked():
            fit_start, fit_end = [10 ** float(self._fit_region.getRegion()[0]), 10 ** float(self._fit_region.getRegion()[1])]
        else:
            fit_start, fit_end = [float(self._fit_region.getRegion()[0]), float(self._fit_region.getRegion()[1])]
        self.statusBar().showMessage("Мин. лазер: {:.2f}\tМакс. лазер: {:.2f}\tСредний: {:.2f}\tСигма: {:.2f}\tОтсеяно: {:.2f}%\tСтарт: {:.5f}\tСтоп: {:.5f}".format(self._laser_min, self._laser_max, self._laser_mean, self._laser_std, self._del_rows_count, fit_start, fit_end))

    def _about(self):
        QtGui.QMessageBox.about(self, "About EDP", "fgsfds")

    def _clear_all(self):
        """Clear all plots and selects"""
        self.laser_plot.clear()
        self._graph_select.clear()
        self._fit_stats = False
        self._fitting_figure.clear()
        self._fitting_stats_figure.clear()
        self.fit_report_text.clear()

    def _open(self):
        from os.path import basename
        filename = QtGui.QFileDialog.getOpenFileName(self, filter="*.dat")
        if filename:
            self.data_filename = filename
            self._clear_all()
            self.data = pd.read_csv(filename, sep='\t', decimal=',')
            self.statusBar().showMessage("File loaded", 2000)
            self.setWindowTitle("EDP - " + basename(filename))
            self._fill_graph_select()
            self._plot_hist()
            self._calc_curve_data()
            # except Exception as e:
            # self.statusBar().showMessage("Loading failed: {}".format(e))

    def _plot_hist(self):
        if not self.data.empty:
            try:
                y, x = np.histogram(self.data[LASER_TITLE], bins=self._bins_widget.value())
                curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(255, 255, 255, 40))
                self.laser_plot.addItem(curve)
                std = self.data[LASER_TITLE].std()
                self._select_region.setRegion([self.data[LASER_TITLE].mean()-std, self.data[LASER_TITLE].mean()+std])
                self.laser_plot.addItem(self._select_region)
            except KeyError:
                self.statusBar().showMessage("Laser field not found in data provided. Nothing to plot.")

    def _create_actions(self):

        self._open_action = QtGui.QAction(QtGui.QIcon(':/images/open.png'),
                                          "&Open...", self, shortcut=QtGui.QKeySequence.Open,
                                          statusTip="Open an existing file", triggered=self._open)

        self._exit_action = QtGui.QAction("E&xit", self,
                                          shortcut=QtGui.QKeySequence.Quit,
                                          statusTip="Exit the application",
                                          triggered=QtGui.qApp.closeAllWindows)

        self._about_action = QtGui.QAction("&About", self,
                                           statusTip="Show the application's About box",
                                           triggered=self._about)

        self._about_qt_action = QtGui.QAction("About &Qt", self,
                                              statusTip="Show the Qt library's About box",
                                              triggered=QtGui.qApp.aboutQt)

        self._fit_action = QtGui.QAction(QtGui.QIcon(':/images/open.png'), "&Fit", self,
                                         statusTip="Fit current data", triggered=self._on_fit_btn_clicked)

        self._reset_initials_action = QtGui.QAction(QtGui.QIcon(':/images/open.png'), "&Reset initials", self,
                                         statusTip="Reset fit starting values to default ones", triggered=self._reset_fit_initials)


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    edp = MainWindow()
    edp.show()
    sys.exit(app.exec_())

