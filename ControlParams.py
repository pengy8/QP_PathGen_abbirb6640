
class ControlParams:
  
    def __init__(self,ex,ey,ez,n,P,H,ttype,dq_bounds,q,dq,pos,orien,pos_v,ang_v, w_t,v_t,epsilon,inc_pos_v,inc_ang_v,stop,er,ep,upper_dq_bounds):
        self.params = self.ControlParams(ex,ey,ez,n,P,H,ttype,dq_bounds,q,dq,pos,orien,pos_v,ang_v,w_t,v_t,epsilon,inc_pos_v,inc_ang_v,stop,er,ep,upper_dq_bounds)
    
        
    def ControlParams(self, ex,ey,ez,n,P,H,ttype,dq_bounds,q,dq,pos,orien,pos_v,ang_v,w_t,v_t,epsilon,inc_pos_v,inc_ang_v,stop,er,ep,upper_dq_bounds):
            defi = {}
            controls = {}
            plots = {}
            keyboard = {}
            opt = {}
            
            defi['ex'] = ex
            defi['ey'] = ey            
            defi['ez'] = ez
            defi['n'] = n
            defi['P'] = P
            defi['H'] = H
            defi['ttype'] = ttype
            defi['dq_bounds'] = dq_bounds
            
            controls['q'] = q
            controls['dq'] = dq
            controls['pos'] = pos
            controls['orien'] = orien
            controls['pos_v'] = pos_v
            controls['ang_v'] = ang_v
            controls['w_t'] = w_t
            controls['v_t'] = v_t
            controls['epsilon'] = epsilon
            controls['stop'] = stop
            
            keyboard['inc_pos_v'] = inc_pos_v 
            keyboard['inc_ang_v'] = inc_ang_v 
            
            opt['er'] = er
            opt['ep'] = ep
            opt['upper_dq_bounds'] = upper_dq_bounds
            
            params = {'defi': defi, 'controls': controls, 'plots': plots, 'keyboard': keyboard, 'opt': opt}
            return params
