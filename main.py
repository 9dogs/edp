# -*- coding: utf-8 -*-
"""
Echo data calculation software
"""
# Major library imports
import numpy as np
import pandas as pd

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


LASER_TITLE = 'Laser'
DELAY_TITLE = 'Time'


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

        self._central_widget = QtGui.QWidget()
        self._main_layout = QtGui.QVBoxLayout()

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

        # Бины для гистограммы лазера
        self._bins_widget = QtGui.QSpinBox()
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

        # Layout для выбора графиков и бинов
        self._aux_layout = QtGui.QHBoxLayout()
        self._aux_layout.addWidget(self._bins_widget)
        self._aux_layout.addWidget(self._graph_select)
        self._aux_layout.addStretch(5)
        self._aux_layout.addWidget(self._copy_data_btn)

        self._main_layout.addWidget(self._plot_area)
        self._main_layout.addLayout(self._aux_layout)

        self._central_widget.setLayout(self._main_layout)
        self.setCentralWidget(self._central_widget)

        self._create_actions()
        self._create_menu()
        self._create_status_bar()

    def _on_v_line_pos_changed(self):
        self._calc_curve_data()

    def _on_copy_data_clicked(self):
        if not self.averaged_data.empty:
            self.averaged_data.to_clipboard()

    def _update_line_pos(self):
        try:
            self._v_line.setPos(self.averaged_data.mean())
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

    def _on_bins_value_changed(self):
        """
        Функция срабатывает при изменении бинов
        """
        if not self.data.empty:
            y, x = np.histogram(self.data[LASER_TITLE], bins=self._bins_widget.value())
            curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(255, 255, 255, 40))
            self.laser_plot.clear()
            self.laser_plot.addItem(curve)
            self._select_region.setRegion([self.data[LASER_TITLE].min(), self.data[LASER_TITLE].max()])
            self.laser_plot.addItem(self._select_region)

    def _create_menu(self):
        self._file_menu = self.menuBar().addMenu("&File")
        self._file_menu.addAction(self._open_action)
        self._file_menu.addAction(self._exit_action)

        self._help_menu = self.menuBar().addMenu("&Help")
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def _calc_curve_data(self):
        """
        Calculates data for echo plot
        """
        laser_min, laser_max = self._select_region.getRegion()
        # good data indices
        good_data = self.data[(laser_min < self.data[LASER_TITLE]) & (self.data[LASER_TITLE] < laser_max)]
        mean = good_data[LASER_TITLE].mean()
        std = good_data[LASER_TITLE].std()

        self.statusBar().showMessage(
            "Мин. лазер: {:.2f}\tМакс. лазер: {:.2f}\tСредний: {:.2f}\tСигма: {:.2f}\tСтарт: {:.5f}".format(laser_min,
                                                                                                            laser_max,
                                                                                                            mean,
                                                                                                            std,
                                                                                                            self._v_line.value()))

        self.averaged_data = good_data.groupby(DELAY_TITLE).mean().reset_index()
        self.echo_plot.clear()
        self.echo_plot.plot(self.averaged_data[DELAY_TITLE], self.averaged_data[self._graph_select.currentText()])
        self.echo_plot.addItem(self._v_line)
        self._update_line_pos()

    def _about(self):
        QtGui.QMessageBox.about(self, "About EDP", "fgsfds")

    def _open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, filter="*.dat")
        if filename:
            self.data = pd.read_csv(filename, sep='\t', decimal=',')
            self.statusBar().showMessage("File loaded", 2000)
            self._fill_graph_select()
            self._plot_hist()
            # except Exception as e:
            # self.statusBar().showMessage("Loading failed: {}".format(e))

    def _plot_hist(self):
        if not self.data.empty:
            try:
                y, x = np.histogram(self.data[LASER_TITLE], bins=self._bins_widget.value())
                curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(255, 255, 255, 40))
                self.laser_plot.addItem(curve)
                self._select_region.setRegion([self.data[LASER_TITLE].min(), self.data[LASER_TITLE].max()])
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


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    edp = MainWindow()
    edp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

