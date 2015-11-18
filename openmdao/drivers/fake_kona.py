
from six import iteritems

from openmdao.core.driver import Driver
from openmdao.util.string_util import nearest_child

class FakeKona(Driver):

    def _setup(self, root):
        super(FakeKona, self)._setup(root)

        obj_cnames, self._obj_comps = self._comps_from_vars(self._objs)
        con_cnames, self._con_comps = self._comps_from_vars(self._cons)

        self._obj_xfers = []
        self._con_xfers = []
        for grp in self.root.subgroups(recurse=True, include_self=True):
            for cname in obj_cnames:
                tup = (nearest_child(grp.pathname, cname), 'fwd', None)
                if tup in grp._data_xfer:
                    self._obj_xfers.append((grp, grp._data_xfer[tup]))

            for cname in con_cnames:
                tup = (nearest_child(grp.pathname, cname), 'fwd', None)
                if tup in grp._data_xfer:
                    self._con_xfers.append((grp, grp._data_xfer[tup]))

    def quick_objective_eval(self):
        for grp, xfer in self._obj_xfers:
            #print("scatter",xfer.src_idxs, "-->", xfer.tgt_idxs)
            xfer.transfer(grp.unknowns, grp.params, 'fwd')
        for comp in self._obj_comps:
            #print("evaluating",comp.pathname)
            comp.solve_nonlinear(comp.params, comp.unknowns, comp.resids)

    def quick_constraint_eval(self):
        for grp, xfer in self._con_xfers:
            #print("scatter",xfer.src_idxs, "-->", xfer.tgt_idxs)
            xfer.transfer(grp.unknowns, grp.params, 'fwd')
        for comp in self._con_comps:
            #print("evaluating",comp.pathname)
            comp.solve_nonlinear(comp.params, comp.unknowns, comp.resids)

    def _comps_from_vars(self, names):
        abs_u = self.root._sysdata.to_abs_uname

        cnames = set(abs_u[n].rsplit('.', 1)[0] for n in names)
        return cnames, [s for s in self.root.components(recurse=True)
                          if s.pathname in cnames]
