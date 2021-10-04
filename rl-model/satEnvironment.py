# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:14:48 2021

@author: sgboakes
"""

class satEnvironment(object):
    def reset(self):
        '''Return initial_time_step'''
        self._current_time_step = self._reset()
        return self._current_time_step
    
    def step(self, action):
        '''Apply action and return new time step'''
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step
    
    def current_time_step(self):
        return self._current_time_step
    
    def time_step_spec(self):
        '''Return time_step_spec'''
        
    @abc.abstractmethod
    def observation_spec(self):
        '''Return observation_spec'''
        
    @abc.abstractmethod
    def action_spec(self):
        '''Return action_spec'''
        
    @abc.abstractmethod
    def _reset(self):
        '''Return initial_time_step'''
        
    @abc.abstractmethod
    def _step(self, action):
        '''Apply action and return new time_step'''