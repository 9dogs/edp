# -*- coding: utf-8 -*-
"""
Echo data calculation software
"""
# Major library imports
from numpy import histogram
import time


# Enthought library imports
from enable.api import ComponentEditor
from traits.api import HasTraits, Float, Range, Button, Enum, File
from traitsui.api import Item, View, HGroup, spring, RangeEditor
from enthought.traits.ui.key_bindings import KeyBinding, KeyBindings

# Chaco imports
from chaco.api import ArrayPlotData, OverlayPlotContainer, Plot, PlotAxis, PlotGrid
from chaco.tools.api import RangeSelection, RangeSelectionOverlay, PanTool, ZoomTool


def follow(thefile):
    """
    Follows the file like tail -f
    """
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


def find(target_list, val):
    """
    Finds all the positions of val in list
    """
    return [i for i, element in enumerate(target_list) if element == val]


class Window(HasTraits):

    calc_button = Button("Calculate")
    copy_button = Button("Copy Data")
    refresh_button = Button("Refresh")
    graph_list = Enum('delay', 'temp', 'npe1', 'npe2', 'npe3', 'laser', 'extra')

    stats = []

    step = Range(1., 5000.)
    low = Float(1.)
    high = Float(5000.)

    xmin = Float
    xmax = Float
    laser_min = Float
    laser_max = Float

    hit_rate = Float
    av_num = Float

    file_name = File

    laserdata = ArrayPlotData()
    echodata = ArrayPlotData()

    bins = Range(30, 400)

    data = {'delay': [], 'temp': [], 'npe1': [], 'npe2': [],
            'npe3': [], 'laser': [], 'extra': []}

    size = (800, 300)
    bar_container = OverlayPlotContainer(bgcolor="white", use_backbuffer=True)
    line_container = OverlayPlotContainer(bgcolor="white", use_backbuffer=True)

    # Key bindings
    move_right_bind = KeyBinding(binding1='.', binding2='>', method_name='move_right')
    move_left_bind = KeyBinding(binding1=',', binding2='<', method_name='move_left')
    calc_bind = KeyBinding(binding1='m', binding2='ь', method_name='calc_bind_fired')
    bindings = KeyBindings(move_left_bind, move_right_bind, calc_bind)

    # TraitsUI definition
    traits_view = View(HGroup(Item('refresh_button', show_label=False),
                              Item('file_name', style='simple',
                                                label='File'),
                              spring,
                              Item('step', label='Scan step',
                                    editor=RangeEditor(mode='auto',
                                                       low_name='low',
                                                       high_name='high')
                                   )
                              ),
                       Item('bar_container',
                            editor=ComponentEditor(size=size),
                            show_label=False, resizable=True),
                       Item('line_container',
                            editor=ComponentEditor(size=size),
                            show_label=False, resizable=True),
                       HGroup(Item("bins"),
                              Item('graph_list'), spring,
                              Item("laser_min", style='readonly'), spring,
                              Item("laser_max", style='readonly'), spring,
                              Item("hit_rate", style='readonly'), spring,
                              Item('av_num', style='readonly'), spring,
                              show_border=True, label="Info"),
                       HGroup(spring, Item('calc_button', show_label=False),
                                      Item('copy_button', show_label=False)),
                        key_bindings=bindings)

    def __init__(self):
        super(Window, self).__init__()
        self.bins = 100
        self._create_plot()

    def _file_name_changed(self, filename):
        """
        Open file button handler
        """
        filename = filename.replace('\\', '/')
        self.data = {'delay': [], 'temp': [], 'npe1': [], 'npe2': [],
                     'npe3': [], 'laser': [], 'extra': []}
        self._read_data(filename)
        index, value = self._calc_data(self.bins)
        self._fill_plot(self.laser_plot, index, value)

    def _create_plot(self):
        """
        Creates 2 plots with dummy data
        """
        # Prepare initial data for Plot
        index, value = [0,], [0,]
        # print index, value
        self.laserdata.set_data('Laser', index)
        self.laserdata.set_data('Counts', value)

        # Create Plot object
        self.laser_plotter = Plot(data=self.laserdata)

        # Select subplot to work with
        self.laser_plot = self.laser_plotter.plot(['Laser', 'Counts'],
                                              type='line')[0]

        ############## Add tools ###########
        # Pan Tool
        self.laser_plot.tools.append(PanTool(self.laser_plot, constrain=True,
                                       constrain_direction="x"))

        # Zoom Tool
        self.laser_plot.overlays.append(ZoomTool(self.laser_plot,
                                                 always_on=False,
                                                 tool_mode = "box",
                                                 axis = "index",
                                                 max_zoom_out_factor = 1.0))

        # Range Select Tool
        self.rangeselect = RangeSelection(component=self.laser_plot,
                                          left_button_selects=False,
                                          auto_handle_event=False)
        self.laser_plot.active_tool = self.rangeselect
        self.laser_plot.overlays.append(RangeSelectionOverlay(component=\
                                        self.laser_plot))
        self.rangeselect.on_trait_change(self.on_selection_changed,
                                         "selection")
        ############## [Add tools] ###########

        # Configure axes and grids
        self._configure_plot(self.laser_plot)

        # add plot to the container
        self.bar_container.add(self.laser_plot)


        # Create Plot object
        self.echodata.set_data('Delay', [0,])
        self.echodata.set_data('Echo', [0,])
        self.echo_plotter = Plot(data=self.echodata)

        # Select subplot to work with
        self.echo_plot = self.echo_plotter.plot(['Delay', 'Echo'],
                                                type='scatter')[0]

        ############## Add tools ###########
        # Pan Tool
        self.echo_plot.tools.append(PanTool(self.echo_plot, constrain=True,
                                       constrain_direction="x"))

        # Zoom Tool
        self.echo_plot.overlays.append(ZoomTool(self.echo_plot, always_on=False,
                                           tool_mode = "box",
                                           axis = "index",
                                           max_zoom_out_factor = 1.0))

        ############## [Add tools] ###########

        # Configure axes and grids
        self._configure_plot(self.echo_plot)

        # add plot to the container
        self.line_container.add(self.echo_plot)


    def _fill_plot(self, plot, xdata, ydata):
        """
        Fill plot with data and request for redraw
        """
        self.laserdata.set_data('Laser', xdata)
        self.laserdata.set_data('Counts', ydata)
        plot.request_redraw()


    def _fill_echo_plot(self, plot, xdata, ydata):
        """
        Fill plot with data and request for redraw
        """
        self.echodata.set_data('Delay', xdata)
        self.echodata.set_data('Echo', ydata)
        plot.request_redraw()


    def _calc_data(self, bins):
        """
        Return histogram data with `bins` number of bins
        """
        hist_data, bin_edges = histogram(self.data['laser'], bins=bins)
        xdata = []

        for i in range(bins):
            xdata.append((bin_edges[i] + bin_edges[i+1]) / 2.)

        return (xdata, list(hist_data))


    def on_selection_changed(self, selection):
        """
        Handler for RectangleSelection tool change event
        """
        if selection != None:
            self.laser_min, self.laser_max = selection
            # self._calc_curve_data(self.laser_min, self.laser_max)


    def _read_data(self, filename):
        """
        Read data from file and fill self.data variable
        """
        with open(filename, 'r') as datafile:
            # skip a header line
            next(datafile)
            for lineno, line in enumerate(datafile):
                # replace all commas with dots and delete whitespaces
                # try to parse string
                try:
                    delay, temp, npe1, npe2, npe3, laser, extra, datetime = \
                    line.replace(',', '.').rstrip().split('\t')
                    self.data['delay'].append(float(delay))
                    self.data['temp'].append(float(temp))
                    self.data['npe1'].append(float(npe1))
                    self.data['npe2'].append(float(npe2))
                    self.data['npe3'].append(float(npe3))
                    self.data['laser'].append(float(laser))
                    self.data['extra'].append(float(extra))
                except Exception as ex:
                    print("Can't split string # %s: %s" % (lineno, ex))


    def _bins_changed(self):
        """
        Handler for bins changed event
        """
        x, y = self._calc_data(self.bins)
        self._fill_plot(self.laser_plot, x, y)


    def _configure_plot(self, plot, xlabel='Laser'):
        """
        Set up colors, grids, etc. on plot objects.
        """
        plot.bgcolor='white'
        plot.border_visible=True
        plot.padding=[40, 15, 15, 20]
        plot.color='darkred'
        plot.line_width = 1.1

        vertical_grid = PlotGrid(component=plot,
                                 mapper=plot.index_mapper,
                                 orientation='vertical',
                                 line_color="gray",
                                 line_style='dot',
                                 use_draw_order=True)

        horizontal_grid = PlotGrid(component=plot,
                                   mapper=plot.value_mapper,
                                   orientation='horizontal',
                                   line_color="gray",
                                   line_style='dot',
                                   use_draw_order=True)

        vertical_axis = PlotAxis(orientation='left',
                                 mapper=plot.value_mapper,
                                 use_draw_order=True)

        horizontal_axis = PlotAxis(orientation='bottom',
                                  title=xlabel,
                                  mapper=plot.index_mapper,
                                  use_draw_order=True)

        # plot.underlays.append(vertical_grid)
        # plot.underlays.append(horizontal_grid)

        # Have to add axes to overlays because we are backbuffering the main
        # plot, and only overlays get to render in addition to the backbuffer.
        plot.overlays.append(vertical_axis)
        plot.overlays.append(horizontal_axis)


    def _calc_curve_data(self, laser_min, laser_max):
        """
        Calculates data for echo plot
        """
        # good data indices
        good_indices = []
        for index, laser in enumerate(self.data['laser']):
            if laser > laser_min and laser < laser_max:
                good_indices.append(index)
        self.hit_rate = float(len(good_indices)) / \
                        float(len(self.data['laser'])) * 100.
        good_data = {'delay': [], 'temp': [], 'npe1': [], 'npe2': [],
            'npe3': [], 'laser': [], 'extra': []}
        for column in good_data:
            for index in good_indices:
                good_data[column].append(self.data[column][index])
        # print good_data
        self.avereged_data = self._average_data(good_data, 'delay')
        # print self.avereged_data
        self._fill_echo_plot(self.echo_plot, self.avereged_data['delay'],
                             self.avereged_data[self.graph_list])
        # self._configure_plot(self.echo_plot)
        self.av_num = float(sum(self.stats))/float(len(self.stats))


    def _average_data(self, data, column):
        """
        Average all `data` by values in `column` column
        """
        averaged_data = {'delay': [], 'temp': [], 'npe1': [], 'npe2': [],
            'npe3': [], 'laser': [], 'extra': []}
        self.stats = []
        for element in data[column]:
            # print "Element", element
            indices = find(data[column], element)
            self.stats.append(len(indices))
            # print "Found on", indices
            if len(indices)>1:
                for col_name, col_values in data.iteritems():
                    av_data = self._average_column(col_values, indices)
                    self._clean_data(col_values, indices)
                    #print averaged_data
                    averaged_data[col_name].append(av_data)

        return averaged_data

    def _average_column(self, column, indices):
        """
        Returns average of all elements in list `column`
        with given list of indices `indices`
        """
        l = [ column[i] for i in indices ]
        return float(sum(l))/len(l)


    def _clean_data(self, data, indices):
        """
        Removes elements from list `data` with given list of indices
        `indices`
        """
        data[:] = [ item for i, item in enumerate(data) if i not in indices ]


    def _calc_button_fired(self):
        """
        Handler for CalcButton clicked event
        """
        self._calc_curve_data(self.laser_min, self.laser_max)


    def _refresh_button_fired(self):
        """
        Handler for CalcButton clicked event
        """
        self._file_name_changed(self.file_name)
        self._calc_button_fired()


    def _copy_button_fired(self):
        """
        Handler for CopyData Button clicked event
        """

        header_string = '\t'.join([key for key in self.avereged_data.keys()])
        header_string += '\r\n'
        data_string = ''

        for index in range(len(self.avereged_data['delay'])):
            for column_values in self.avereged_data.values():
                data_string += str(column_values[index]) + '\t'
            data_string += '\r\n'

        import wx

        data_obj = wx.TextDataObject(header_string + data_string)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(data_obj)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard.", "Error")


    def laser_window_changed(self):
        """
        Handler for laser_min and laser_max changes
        """
        self.rangeselect.selection = self.laser_min, self.laser_max

    def move_right(self, info):
        """
        Handler for '>' key click
        """
        self.rangeselect.selection = self.laser_min + self.step,\
                                     self.laser_max + self.step
        self._calc_button_fired()


    def move_left(self, info):
        """
        Handler for '<' key click
        """
        self.rangeselect.selection = self.laser_min - self.step,\
                                     self.laser_max - self.step
        self._calc_button_fired()

    def calc_bind_fired(self, info):
        """
        Handler for 'm' key click
        """
        self._calc_button_fired()


if __name__ == '__main__':
    window = Window()
    window.configure_traits()
