"""
Run R TSdist package for time series metrics.

See:
- https://github.com/Ohohcakester/Charting/blob/master/rbind.py
- http://dtw.r-forge.r-project.org/
- http://rpy.sourceforge.net/rpy2/doc-2.5/html/robjects_rinstance.html
- https://nipunbatra.wordpress.com/2013/06/09/dynamic-time-warping-using-rpy-and-python/
"""
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
r = rpy2.robjects.r
ts = importr("TSdist")


def run_ts(data1, data2, measureName, *args, **kwargs):
    measureName = measureName.replace(".", "_")
    measureFun = getattr(ts, measureName)
    d1 = rpy2.robjects.FloatVector(data1)
    d2 = rpy2.robjects.FloatVector(data2)
    result = measureFun(d1, d2, *args, **kwargs)
    return result[0]


# Gets a measure from the r TSdist library.
# Example: r('dtwDistance') returns a dtw distance similarity measure.
def tsdist(measureName, *args, **kwargs):
    def fun(data1, data2):
        return run_ts(data1, data2, measureName, *args, **kwargs)

    return fun


if __name__ == "__main__":
    pass
