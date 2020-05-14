# -*- coding: utf-8 -*-

__all__ = ["TDVOrbit"]

import numpy as np
import theano.tensor as tt

from .simple import SimpleTransitOrbit


class TDVOrbit(SimpleTransitOrbit):
    """A generalization of a Simple orbit with transit duration variations

    Only one of the arguments ``tdvs`` or ``transit_durs`` can be given and
    the other will be computed from the one that was provided.

    Args:    
        tdvs: 
        
        transit_durs: 
        
        transit_inds: 

    """

    def __init__(self, *args, **kwargs):
        tdvs = kwargs.pop("tdvs", None)
        transit_durs = kwargs.pop("transit_durs", None)
        transit_inds = kwargs.pop("transit_inds", None)
        
        if tdvs is None and transit_durs is None:
            raise ValueError(
                "one of 'tdvs' or 'transit_durs' must be " "defined"
            )
        
        # this is for if tdvs are given
        if tdvs is not None:
            if "duration" not in kwargs:
                raise ValueError("if 'tdvs' is given, 'duration' must also be supplied")
                
            self.tdvs = [tt.as_tensor_variable(tdv, ndim=1) for tdv in tdvs]
                        
            if transit_inds is None:
                self.transit_inds = [
                    tt.arange(tdv.shape[0]) for tdv in self.tdvs
                ]
            else:
                self.transit_inds = [
                    tt.cast(tt.as_tensor_variable(inds, ndim=1), "int64")
                    for inds in transit_inds
                ]
                
            
        # this is for if transit_durs are given
        else:
            if "duration" in kwargs:
                warnings.warn("supplied duration will be overwritten with mean of transit_durs")

            self.transit_durs = [tt.as_tensor_variable(durs, ndim=1) for durs in transit_durs]

            if transit_inds is None:
                self.transit_inds = [
                    tt.arange(dur.shape[0]) for dur in self.transit_durs
                ]
            else:
                self.transit_inds = [
                    tt.cast(tt.as_tensor_variable(inds, ndim=1), "int64")
                    for inds in transit_inds
                ]
            
            duration = []
            for i, durs in enumerate(transit_durs):
                duration.append(tt.mean(durs))
                                
            kwargs["duration"] = tt.stack(duration)
            
            
        # this line allows TDVOrbit to inherit from SimpleTransitOrbit
        super(TDVOrbit, self).__init__(*args, **kwargs)

        # and now we can calculate transit_durs since duration has been inherited
        if tdvs is not None:
            self.transit_durs = [
                self.duration[i] + tdv
                for i, tdv in enumerate(self.tdvs)
            ]
            
        
        else:
            self.tdvs = [
                durs - self.duration[i] 
                for i, durs in enumerate(self.transit_durs)
            ]

