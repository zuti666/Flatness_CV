"""
Compatibility shim: exposes flatness/feature evaluation modules under the
`eval_flat` namespace while reusing the existing implementations in
`evaluation_sharpness`.
"""
from evaluation_sharpness.eval_flatness_weight_Loss import *  # noqa
from evaluation_sharpness.eval_flat_feature import *  # noqa
from evaluation_sharpness.loss_landscape import *  # noqa
