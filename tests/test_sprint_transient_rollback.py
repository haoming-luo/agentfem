import numpy as np
import pytest
from test_transient_restart import _heat_step

def test_failed_load_update_restores_fields_and_time():
    step=_heat_step();before=step.current.x.array.copy();old=step.previous.x.array.copy();times=[]
    def update(t):
        times.append(t)
        if t>0:
            step.current.x.array[:]=999.
            raise RuntimeError('injected load failure')
    step.update_load=update
    with pytest.raises(RuntimeError,match='injected'):step.run()
    assert times[-1]==0.
    assert step.completed_steps==0
    np.testing.assert_array_equal(step.current.x.array,before)
    np.testing.assert_array_equal(step.previous.x.array,old)
    step.update_load=None
    step.run()
    reference=_heat_step();reference.run()
    np.testing.assert_allclose(step.current.x.array,reference.current.x.array)
