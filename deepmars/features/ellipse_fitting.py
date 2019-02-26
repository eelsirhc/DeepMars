from fiterror import FitError
import ellipses as el

def proc_ellipse(fm, roi):
    """Process an image stencil to an ellipse fitted dataset.

    Flipping x,y axes for the image."""
    try:
        data = np.where(fm)
        lsqe = el.LSqEllipse()
        lsqe.fit(data)
        center, width, height, phi = lsqe.parameters()
        res = [center[1], center[0], width*2, 2*height, np.rad2deg(phi)]
        return res
    except Exception as e:
        raise FitError(e)
