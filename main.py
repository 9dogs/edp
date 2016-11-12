# -*- coding: utf-8 -*-
"""
Echo data calculation software
"""
# Major library imports
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets, QtGui

from os.path import basename
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from lmfit.models import LorentzianModel, ConstantModel, Model, GaussianModel
from lmfit.parameter import Parameters, Parameter

from fitting.fit import fit_df, echo_decay_curve, prepare_data, PEAK_MODELS

import seaborn as sns
sns.set(style="whitegrid")


LASER_TITLE = 'laser'
DELAY_TITLE = 'time'


class ParamWidget(QtWidgets.QWidget):
    def __init__(self, name, display_name=None, val=0, vary=True, min=-np.inf, max=np.inf):
        super().__init__(parent=None, flags=QtCore.Qt.Widget)
        self._name = name
        self._display_name = display_name if display_name else name
        self._min = min
        self._max = max
        self._val = val
        self.vary = vary
        self.param = Parameter(name, val, vary, min, max)

        self._label = QtWidgets.QLabel(self._display_name)
        self._min_edit = QtWidgets.QDoubleSpinBox(self)
        self._max_edit = QtWidgets.QDoubleSpinBox(self)
        self._val_edit = QtWidgets.QDoubleSpinBox(self)

        self._set_spinboxes([self._min_edit, self._max_edit, self._val_edit])

        self._vary_checkbox = QtWidgets.QCheckBox(self)
        self._vary_checkbox.setChecked(True)
        self._vary_checkbox.setStyleSheet("""
             QCheckBox::indicator {
                 width: 16px;
                 height: 16px;
             }
            QCheckBox::indicator:checked
              {
                image: url(resources/icons/unlocked.png);
              }
              QCheckBox::indicator:unchecked
              {
                image: url(resources/icons/locked.png);
              }
        """)
        self._vary_checkbox.stateChanged.connect(self._update_parameter)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(QtWidgets.QLabel('('))
        layout.addWidget(self._min_edit)
        layout.addWidget(QtWidgets.QLabel(','))
        layout.addWidget(self._max_edit)
        layout.addWidget(QtWidgets.QLabel(');'))
        layout.addWidget(self._val_edit)
        layout.addWidget(self._vary_checkbox)

        self.setMaximumWidth(250)

    def _set_spinboxes(self, spinboxes):
        for spin in spinboxes:
            spin.setMaximum(99999)
            spin.setMinimum(-99999)
            spin.setSingleStep(1)
            spin.setButtonSymbols(2)
            spin.valueChanged.connect(self._update_parameter)

    def _update_parameter(self):
        val = self._val_edit.value()
        min_val = self._min_edit.value()
        max_val = self._max_edit.value()
        vary = True if self._vary_checkbox.isChecked() else False
        try:
            self.param = Parameter(self._name, val, vary, min_val, max_val)
        except ValueError as e:
            self.param = Parameter(self._name, val, vary)
            #self.statusBar().showMessage("{}, min and max keep unchanged".format(e))

    def from_res(self, res):
        val = res.best_values.get(self._name)
        self._val_edit.setValue(val)


    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, value):
        self._val = value
        self._val_edit.setValue(value)
        self._update_parameter()


class MainWindow(QtWidgets.QMainWindow):
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

        self._central_widget = QtWidgets.QWidget()

        self._graph_tabs_widget = QtWidgets.QTabWidget()
        self._graph_tabs_widget.currentChanged.connect(self._on_tab_changed)

        self._main_layout = QtWidgets.QHBoxLayout()
        self._pyqtgraph_layout = QtWidgets.QVBoxLayout()

        # GraphLayoutWidget с двумя графиками: гистограмма лазера и эхо
        self._plot_area = pg.GraphicsLayoutWidget()
        self.laser_plot = self._plot_area.addPlot()
        self._plot_area.nextRow()
        self.echo_plot = self._plot_area.addPlot()

        # ComboBox для выбора графика
        self._graph_select = QtWidgets.QComboBox()
        self._graph_select.currentIndexChanged.connect(self._calc_curve_data)

        # ComboBox для выбора модели фита
        self._peak_model_select = QtWidgets.QComboBox()
        self._peak_model_select.addItems(PEAK_MODELS.keys())
        self._peak_model_select.addItem('Echo')

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
        fitting_tab = QtWidgets.QWidget()
        # a figure instance to plot on
        self._fitting_figure = plt.figure()
        self._fitting_canvas = FigureCanvas(self._fitting_figure)
        self._ax = self._fitting_figure.add_subplot(111)
        # toolbar
        fitting_toolbar = NavigationToolbar(self._fitting_canvas, self)
        # fit parameters
        params_layout = QtWidgets.QGridLayout()

        # Echo params
        self.y0 = ParamWidget('y0')
        self.A = ParamWidget('A')
        self.t2 = ParamWidget('t2')
        # Initial guesses
        self.y0.value = 5000
        self.A.value = 500
        self.t2.value = 500

        # Peak params
        self.amp = ParamWidget('peak_amplitude', 'Amp')
        self.center = ParamWidget('peak_center', 'Center')
        self.sigma = ParamWidget('peak_sigma', 'Sigma')
        self.c = ParamWidget('const_c', 'Const')
        # Initial guesses
        self.amp.value = 10000
        self.center.value = 0
        self.sigma.value = 5
        self.c.value = 5000

        params_layout.addWidget(self.amp, 0, 0)
        params_layout.addWidget(self.center, 1, 0)
        params_layout.addWidget(self.sigma, 2, 0)
        params_layout.addWidget(self.c, 3, 0)

        params_layout.addWidget(self.y0, 0, 1)
        params_layout.addWidget(self.A, 1, 1)
        params_layout.addWidget(self.t2, 2, 1)

        params_layout.setVerticalSpacing(0)

        fit_layout = QtWidgets.QVBoxLayout()
        fit_layout.addWidget(fitting_toolbar, 2)
        fit_layout.addWidget(self._fitting_canvas, 6)
        fit_layout.addLayout(params_layout, 1)
        fitting_tab.setLayout(fit_layout)

        self._graph_tabs_widget.addTab(fitting_tab, 'Фит')

        # График зависимости t2 от точки старта фита
        fitting_stats_tab = QtWidgets.QWidget()
        # a figure instance to plot on
        self._fitting_stats_figure = plt.figure()
        self._fitting_stats_canvas = FigureCanvas(self._fitting_stats_figure)
        fitting_stats_toolbar = NavigationToolbar(self._fitting_stats_canvas, self)

        fit_stats_layout = QtWidgets.QVBoxLayout()
        fit_stats_layout.addWidget(fitting_stats_toolbar)
        fit_stats_layout.addWidget(self._fitting_stats_canvas)
        fitting_stats_tab.setLayout(fit_stats_layout)

        self._graph_tabs_widget.addTab(fitting_stats_tab, 'Зависимость фита')

        # Фит репорт
        fit_report_widget = QtWidgets.QWidget()
        report_layout = QtWidgets.QVBoxLayout()
        self.fit_report_text = QtWidgets.QTextEdit()
        save_report_btn = QtWidgets.QPushButton("Сохранить")
        save_report_btn.clicked.connect(self._on_save_report_btn_clicked)
        report_layout.addWidget(self.fit_report_text)
        report_layout.addWidget(save_report_btn)
        fit_report_widget.setLayout(report_layout)
        self._graph_tabs_widget.addTab(fit_report_widget, 'Отчет')

        # Бины для гистограммы лазера
        self._bins_widget = QtWidgets.QDoubleSpinBox()
        self._bins_widget.setMinimum(1)
        self._bins_widget.setMaximum(1000)
        self._bins_widget.setValue(200)
        self._bins_widget.valueChanged.connect(self._on_bins_value_changed)

        # Вертикальная линия для фита
        self._v_line = pg.InfiniteLine(angle=90, movable=True)
        self._v_line.sigPositionChangeFinished.connect(self._on_v_line_pos_changed)

        # Кпопка копирования в буфер
        self._copy_data_btn = QtWidgets.QPushButton("Скопировать в буфер")
        self._copy_data_btn.clicked.connect(self._on_copy_data_clicked)

        # Кнопка фита
        self._fit_btn = QtWidgets.QPushButton("Фитировать")
        self._fit_btn.clicked.connect(self._on_fit_btn_clicked)
        self._fit_btn.setToolTip("Shift+Click to calc t2-start fit dependency")

        # Галочка логарифмического масштаба
        self._log_checkbox = QtWidgets.QCheckBox("Лог. масштаб")
        self._log_checkbox.stateChanged.connect(self._on_log_scale_changed)

        # Галочка "нормировать на лазер"
        self._div_laser_checkbox = QtWidgets.QCheckBox("Норм. на лазер")
        self._div_laser_checkbox.stateChanged.connect(self._div_laser_changed)

        # Layout для выбора графиков и бинов
        self._aux_layout = QtWidgets.QHBoxLayout()
        self._aux_layout.addWidget(self._bins_widget)
        self._aux_layout.addWidget(self._graph_select)
        self._aux_layout.addWidget(self._log_checkbox)
        self._aux_layout.addWidget(self._div_laser_checkbox)
        self._aux_layout.addStretch(5)
        self._aux_layout.addWidget(self._peak_model_select)
        self._aux_layout.addWidget(self._fit_btn)
        self._aux_layout.addWidget(self._copy_data_btn)

        self._pyqtgraph_layout.addWidget(self._plot_area)
        self._pyqtgraph_layout.addLayout(self._aux_layout)

        # Установка лэйаутов
        self._splitter = QtWidgets.QSplitter(self)
        w = QtWidgets.QWidget()
        w.setLayout(self._pyqtgraph_layout)
        self._splitter.addWidget(w)
        self._splitter.addWidget(self._graph_tabs_widget)
        # self._main_layout.addLayout(self._pyqtgraph_layout)
        # self._main_layout.addWidget(self._graph_tabs_widget)
        # self._central_widget.setLayout(self._main_layout)
        # self.setCentralWidget(self._central_widget)
        self.setCentralWidget(self._splitter)

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

            model = Model(echo_decay_curve, independent_vars=['x'])
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

    def _fit_echo(self):
        # clean plot
        # self._fitting_figure.clear()

        # Check if x scale is log
        if self._log_checkbox.isChecked():
            fit_start, fit_end = [10 ** float(self._fit_region.getRegion()[0]),
                                  10 ** float(self._fit_region.getRegion()[1])]
        else:
            fit_start, fit_end = [float(self._fit_region.getRegion()[0]), float(self._fit_region.getRegion()[1])]

        # Fit arrays
        all_x = np.array(self.data.index.get_values())
        x = np.array(self.data_in_range.index.get_values())
        if 'res' in self.data_in_range:
            y = self.data_in_range['res']
        else:
            y = self.data_in_range[self._graph_select.currentText() + '_mean']

        params = Parameters()
        params.add_many(
            ('y0', self.y0.param.value, self.y0.param.vary, self.y0.param.min, self.y0.param.max),
            ('A', self.A.param.value, self.A.param.vary, self.A.param.min, self.center.param.max),
            ('t2', self.t2.param.value, self.t2.param.vary, self.t2.param.min, self.t2.param.max),
        )
        model = Model(echo_decay_curve, independent_vars=['x'])

        result = model.fit(y, x=x, params=params)

        time = np.arange(fit_start, fit_end, 0.001)
        all_time = np.linspace(x.min(), x.max(), 2000)

        # plot data
        self._ax.plot(time, echo_decay_curve(x=time, **result.values), '-r', alpha=0.9, linewidth=2)
        self._ax.plot(all_time, echo_decay_curve(x=all_time, **result.values), '--k', alpha=0.4)

        # Подсчитываем данные
        t2 = result.values['t2']
        t2_stderr = result.params.get('t2').stderr
        # Ширина линии в MHz
        delta_f = 10**6/(np.pi*result.values['t2'])
        delta_f_stderr = delta_f*t2_stderr/t2
        self._ax.set_title("$T_2$: {:0.2f} $\pm$ {:0.2f} $ps$ ({:0.2f} $\pm$ {:0.2f} $MHz$ )".format(t2, t2_stderr,
                                                                                               delta_f, delta_f_stderr))
        # refresh canvas and tight layout
        self._fitting_canvas.draw()
        self._fitting_figure.tight_layout()
        # Populating report
        self.fit_report_text.append(result.fit_report())
        # Populating spinboxes
        self.y0.from_res(result)
        self.A.from_res(result)
        self.t2.from_res(result)

    def _fit_peak(self):
        model = PEAK_MODELS[self._peak_model_select.currentText()]

        # Check if x scale is log
        if self._log_checkbox.isChecked():
            fit_start, fit_end = [10 ** float(self._fit_region.getRegion()[0]),
                                  10 ** float(self._fit_region.getRegion()[1])]
        else:
            fit_start, fit_end = [float(self._fit_region.getRegion()[0]), float(self._fit_region.getRegion()[1])]

        #print(fit_start, fit_end, self.data[self._graph_select.currentText()])
        #print(self.data_in_range.head())
        field_name = self._graph_select.currentText() + '_mean'
        params = Parameters()
        params.add_many(
            ('peak_amplitude', self.amp.param.value, self.amp.param.vary, self.amp.param.min, self.amp.param.max),
            ('peak_center', self.center.param.value, self.center.param.vary, self.center.param.min, self.center.param.max),
            ('peak_sigma', self.sigma.param.value, self.sigma.param.vary, self.sigma.param.min, self.sigma.param.max),
            ('const_c', self.c.param.value, self.c.param.vary, self.c.param.min, self.c.param.max),
        )

        df, res = fit_df(self.data_in_range, params=params, model=model, fit_range=(fit_start, fit_end), fit_field=field_name)
        self.amp.from_res(res)
        self.center.from_res(res)
        self.sigma.from_res(res)
        self.c.from_res(res)

        self.data_in_range['res'] = df.peak_fit_res

        x = np.array(df.index.get_values())
        y = np.array(df.peak_fit.values)
        self._fitting_figure.clear()
        self._ax = self._fitting_figure.add_subplot(111)
        # plot data
        self._ax.plot(x, df[field_name], '.')
        # plot fit
        fit_x = np.arange(x.min(), x.max(), 0.001)
        fit_y = res.eval(x=fit_x)
        self._ax.plot(fit_x, fit_y, '-', alpha=0.7, linewidth=2)
        self._ax.plot(x, df.peak_fit_res + self.c.param.value, '.', alpha=0.7)
        self._ax.set_xlim([fit_start, fit_end])
        # refresh canvas and tight layout
        self._fitting_canvas.draw()
        self._fitting_figure.tight_layout()

    def _on_fit_btn_clicked(self):
        if self._peak_model_select.currentText() == 'Echo':
            self._fit_echo()
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ShiftModifier:
                self._calc_fit_start_dep_curve()

        else:
            self._fit_peak()

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
            cols = [x for x in self.data.columns if x != 'time']
            self._graph_select.addItems(cols)

    def _on_region_changed(self):
        pass

    def _reset_fit_initials(self):
        # Initial guesses
        self.y0.value = 5000
        self.A.value = 500
        self.t2.value = 500

        # Initial guesses
        self.amp.value = 10000
        self.center.value = 0
        self.sigma.value = 5
        self.c.value = 5000

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
        self.data_in_range = prepare_data(self.data, laser_window=(self._laser_min, self._laser_max),
                                          laser_title=LASER_TITLE, time_title=DELAY_TITLE)

        selected_graph = self._graph_select.currentText() + '_mean'

        self._del_rows_count = (self.data.shape[0] - self.data_in_range.shape[0])/self.data.shape[0] * 100
        self._laser_mean = self.data_in_range[LASER_TITLE + '_mean'].mean()
        self._laser_std = self.data_in_range[LASER_TITLE + '_mean'].std()

        # Divided by laser
        self.data_in_range['normed'] = self.data_in_range[selected_graph] / self.data_in_range[LASER_TITLE + '_mean']

        self.echo_plot.clear()
        plot_column = 'normed' if self._div_laser_checkbox.isChecked() else selected_graph
        self.echo_plot.plot(self.data_in_range.index.get_values(), self.data_in_range[plot_column], pen=(200, 200, 200), symbolBrush=(230, 0, 0, 0.8*255), symbolPen='w', symbolSize=3)
        err = pg.ErrorBarItem(x=self.data_in_range.index.get_values(), y=self.data_in_range[selected_graph],
                              height=2*self.data_in_range[self._graph_select.currentText() + '_std'], beam=0.5)
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
        QtWidgets.QMessageBox.about(self, "About EDP", "fgsfds")

    def _clear_all(self):
        """Clear all plots and selects"""
        self.laser_plot.clear()
        self._graph_select.clear()
        self._fit_stats = False
        self._fitting_figure.clear()
        self._fitting_stats_figure.clear()
        self.fit_report_text.clear()

    def _open(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, filter="*.dat")
        if filename:
            self.data_filename = filename[0]
            self._clear_all()
            self.data = pd.read_csv(self.data_filename, sep='\t', decimal=',')
            # Pull column name to lower
            self.data.columns = [c.lower() for c in self.data.columns]
            self.statusBar().showMessage("File loaded", 2000)
            self.setWindowTitle("EDP - " + basename(self.data_filename))
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

        self._open_action = QtWidgets.QAction(QtGui.QIcon(':/images/open.png'),
                                          "&Open...", self, shortcut=QtGui.QKeySequence.Open,
                                          statusTip="Open an existing file", triggered=self._open)

        self._exit_action = QtWidgets.QAction("E&xit", self,
                                          shortcut=QtGui.QKeySequence.Quit,
                                          statusTip="Exit the application",
                                          triggered=QtWidgets.qApp.closeAllWindows)

        self._about_action = QtWidgets.QAction("&About", self,
                                           statusTip="Show the application's About box",
                                           triggered=self._about)

        self._about_qt_action = QtWidgets.QAction("About &Qt", self,
                                              statusTip="Show the Qt library's About box",
                                              triggered=QtWidgets.qApp.aboutQt)

        self._fit_action = QtWidgets.QAction(QtGui.QIcon(':/images/open.png'), "&Fit", self,
                                         statusTip="Fit current data", triggered=self._on_fit_btn_clicked)

        self._reset_initials_action = QtWidgets.QAction(QtGui.QIcon(':/images/open.png'), "&Reset initials", self,
                                         statusTip="Reset fit starting values to default ones", triggered=self._reset_fit_initials)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    edp = MainWindow()
    edp.show()
    sys.exit(app.exec_())

