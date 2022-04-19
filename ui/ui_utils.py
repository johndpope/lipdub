import vtk
from custom_types import *



bg_source_color = (152, 181, 234)
bg_target_color = (250, 200, 152)
button_color = (255, 0, 255)
bg_menu_color = (214, 139, 202)
bg_stage_color = (255, 180, 110)
default_colors = [(82, 108, 255), (160, 82, 255), (255, 43, 43), (255, 246, 79),
                  (153, 227, 107), (58, 186, 92), (8, 243, 255), (240, 136, 0)]


RGB_COLOR = Union[Tuple[int, int, int], List[int]]
RGB_FLOAT_COLOR = Union[Tuple[float, float, float], List[float]]
RGBA_COLOR = Union[Tuple[int, int, int, int], List[int]]
RGBA_FLOAT_COLOR = Union[Tuple[float, float, float, float], List[float]]


def channel_to_float(*channel: int):
    if type(channel[0]) is float and 0 <= channel[0] <= 1:
        return channel
    return [c / 255. for c in channel]


def rgb_to_float(*colors: RGB_COLOR) -> Union[RGB_FLOAT_COLOR, List[RGB_FLOAT_COLOR]]:
    float_colors = [channel_to_float(*c) for c in colors]
    if len(float_colors) == 1:
        return float_colors[0]
    return float_colors


def make_slider(iren, observer, render, h, title, val=0.):
    slider_repres = vtk.vtkSliderRepresentation2D()
    slider_repres.SetMinimumValue(0)
    slider_repres.SetMaximumValue(100.)

    slider_repres.SetValue(val)
    slider_repres.GetSliderProperty().SetColor(*rgb_to_float(bg_target_color))
    slider_repres.ShowSliderLabelOff()
    # slider_repres.GetLabelProperty().SetColor(1., 0., 0.)
    slider_repres.GetCapProperty().SetColor(*rgb_to_float(bg_menu_color))
    slider_repres.GetSelectedProperty().SetColor(1., 0., 0)
    slider_repres.GetTubeProperty().SetColor(*rgb_to_float(bg_source_color))
    slider_repres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_repres.GetPoint1Coordinate().SetValue(0.01,  h)
    slider_repres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_repres.GetPoint2Coordinate().SetValue(0.4, h)
    slider_repres.SetSliderLength(0.01)
    slider_repres.SetSliderWidth(0.01)
    slider_repres.SetEndCapLength(0.01)
    slider_repres.SetEndCapWidth(0.01)
    slider_repres.SetTubeWidth(0.01)
    slider_repres.GetTitleProperty().SetColor(0, 0, 0)
    # slider_repres.GetTitleProperty().GetJustification()
    slider_repres.SetTitleText(title)
    slider_repres.SetTitleHeight(0.015)
    # slider_repres.GetLabelProperty().SetVerticalJustification(0)
    slider_repres.SetLabelFormat('%f')
    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetCurrentRenderer(render)
    slider_widget.SetInteractor(iren)
    slider_widget.SetRepresentation(slider_repres)
    slider_widget.KeyPressActivationOff()
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.SetEnabled(True)
    slider_widget.AddObserver('InteractionEvent', observer)
    slider_widget.EnabledOn()

    return slider_widget, slider_repres
