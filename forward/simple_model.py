import numpy as np
import torch

class SimpleLayerModel():
    
    def __init__(self,depths,speeds,source_pos,detector_pos,sample_period,duration,pulse_width):
        
        depths = np.array(depths) 
        self.thicknesses = depths - np.concatenate([np.array([0]),depths[:-1]])
        
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
        
        times = np.zeros((len(self.thicknesses),nd,ns))
        
        amplitudes = np.zeros((nt,nd,ns)) if self.pulse_width else None
        
        xd = self.detector_pos
        
        for (i,xs) in enumerate(self.source_pos):

            for (j,d) in enumerate(self.thicknesses):
            
                d = self.thicknesses[:j+1].reshape(-1,1)
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
                    #A = 2/(np.sqrt(3*a)*(np.pi**0.25))
                    A = 1
                    
                    t = np.tile((np.arange(nt)*self.sample_period).reshape(-1,1),(1,nd))
                    t -= T
                                    
                    layer = A * (1 - (t/a)**2) * np.exp(-0.5*(t/a)**2)
            
                    amplitudes[:,:,i] += layer
                
        
        return times, amplitudes


class SimpleLayerProblem():

    def __init__(self,model,n_samples=200,interval=5,thickness=300,speed=(500,1000),mode='normal',normalize=True):

        # thickness = mean thickness ~ Poisson
        # speed = (low, high) ~ Uniform

        self.model = model

        self.n_samples = n_samples
        self.interval = interval
        self.thickness = thickness
        self.speed = speed
        self.mode = mode
        
        self.normalize = normalize
        
        #self.model.duration = 2*n_samples*interval/speed[0]
        
    def generate_pair(self):
    
        # Generate depths according to Poisson distribution
        
        depth = 0
        speeds = []
        thicknesses = []
        
        speeds_sparse = np.zeros(self.n_samples)
        
        speed = self.speed[1]
        
        mean_n_layers = self.n_samples*self.interval/self.thickness
        mean_max_speed = mean_n_layers*np.mean(self.speed)
        
        while True:
            
            thickness = np.random.exponential(self.thickness)
            speed = speed+np.random.uniform(*self.speed)
            
            speeds_sparse[int(depth/self.interval):] = speed
            
            depth += thickness
                                   
            thicknesses.append(thickness)            
            speeds.append(speed)
            
            if depth > self.n_samples*self.interval:
                break

                   
        # Add direct path
        speeds.insert(0,speeds[0]) 
        thicknesses.insert(0,1) 
        
        
        self.model.thicknesses = np.array(thicknesses)
        self.model.speeds = np.array(speeds)

        times, amplitudes = self.model.propagateSmallAngle() 
        
        if self.normalize:
            speeds_sparse = speeds_sparse/mean_max_speed
        
        if self.mode == 'diff':
            speeds_sparse = np.diff(speeds_sparse,prepend=0)
        
        return amplitudes, speeds_sparse 
    


class SimpleLayerDataset(torch.utils.data.IterableDataset):

    def __init__(self,problem,n_samples=1000):

        self.problem = problem        
        self.n_samples = n_samples
        
    def __len__(self):
        
        return self.n_samples
        
    def __iter__(self):
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_samples = self.n_samples
        else:
            my_samples = self.n_samples // worker_info.num_workers

        for i in range(my_samples):
            
            amplitudes, speeds = self.problem.generate_pair()
            
            # Channels first.
            amplitudes = amplitudes.transpose(2,0,1)            
                        
            yield {'amplitudes': torch.from_numpy(amplitudes), 'speeds': torch.from_numpy(speeds)}
    
    
    
    
    
    
    
    
    
    
    