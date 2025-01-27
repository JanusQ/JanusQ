

import numpy as np

class SPSA:
    def __init__( self, a=3 , c=0.01, A=0, α=0.602 , γ=0.101):
        self._a = a
        self._c = c
        self._A = A
        self._α = α
        self._γ = γ
        self._k = 0
        
    def step( self, function, θ ):
        
        ak = self._a/(self._k+self._A+1)**self._α, 
        ck = self._c/(self._k+1)**self._γ
                
        Δ  = 2*np.round( np.random.rand(θ.size).reshape(θ.shape) )-1
        
        θ_plus  = θ + ck*Δ
        θ_minus = θ - ck*Δ   

        function_plus  = function( θ_plus )  
        function_minus = function( θ_minus )  
        
        ghat = np.divide( function_plus-function_minus, 2*ck*Δ + 1e-8 )
        
        θ  = θ - ak*ghat 
        self._k += 1
        return θ
    
class Adam:
    def __init__( self, α=0.01, β1=0.9, β2=0.999, ϵ=1e-8,c=0.01, γ=0.101):
        self._α = α
        self._β1 = β1
        self._β2 = β2
        self._ϵ = ϵ
        self._m = 0
        self._v = 0
        self._t = 0
        self._c = c
        self._γ = γ
        
    def gradient(self, function, θ):
        ck = self._c/(self._t+1)**self._γ
        Δ  = 2*np.round( np.random.rand(θ.size).reshape(θ.shape) )-1
        
        θ_plus  = θ + ck*Δ
        θ_minus = θ - ck*Δ   

        function_plus  = function( θ_plus )  
        function_minus = function( θ_minus )  
        ghat = np.divide( function_plus-function_minus, 2*ck*Δ + 1e-8 )

        return ghat
    
    def step( self, function, θ):
        self._t += 1
        g = self.gradient(function,θ)
        self._m = self._β1*self._m + (1-self._β1)*g
        self._v = self._β2*self._v + (1-self._β2)*g**2
        mhat = self._m/(1-self._β1**self._t)
        vhat = self._v/(1-self._β2**self._t)
        θ  = θ - self._α*mhat/(np.sqrt(vhat)+self._ϵ)
        return θ
    

    
