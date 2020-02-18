import numpy as np

class SimpleLayerModel():
    
    def __init__(self,depths,speeds,source_pos,detector_pos,sample_period,duration,pulse_width):
        
        self.depths = np.array(depths) 
        self.speeds = np.array(speeds)
        self.source_pos = np.array(source_pos)
        self.detector_pos = np.array(detector_pos) 
        
        self.sample_period = sample_period
        self.duration = duration

        self.pulse_width = pulse_width
        

    def propagateSmallAngle(self):
        
        ns = len(self.source_pos)
        nd = len(self.detector_pos)
        nt = int(self.duration/self.sample_period)
        
        times = np.zeros((len(self.depths),nd,ns))
        
        amplitudes = np.zeros((nt,nd,ns)) if self.pulse_width else None
        
        xd = self.detector_pos
        
        for (i,xs) in enumerate(self.source_pos):

            for (j,d) in enumerate(self.depths):
            
                d = self.depths[:j+1].reshape(-1,1)
                c = self.speeds[:j+1].reshape(-1,1)
                
                D = np.abs(xs - xd).reshape(1,-1)
                
                th = c*0.5*D / np.sum(d*c)
                
                s = d*np.sqrt(1 + th**2)
    
                T = np.sum(s/c,axis=0)*2
            
                times[j,:,i] = T
                
                if self.pulse_width:
                    # Update amplitudes
                    
                    T = T.reshape(1,nd)
                    
                    a = self.pulse_width                
                    A = 2/(np.sqrt(3*a)*(np.pi**0.25))
                    
                    t = np.tile((np.arange(nt)*self.sample_period).reshape(-1,1),(1,nd))
                    t -= T
                                    
                    layer = A * (1 - (t/a)**2) * np.exp(-0.5*(t/a)**2)
            
                    amplitudes[:,:,i] += layer
                
        
        return times, amplitudes

