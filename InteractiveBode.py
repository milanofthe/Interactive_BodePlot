#imports
import numpy as np
import pygame
import os
from time import time


# MISC =====================================================================


def timer(func):
    
    """
    This function shows the execution time 
    of the function object passed
    
    """
    
    def wrap_func(*args, **kwargs):
        
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        
        print(f'Function {func.__name__!r} executed in {round((t2-t1)*1e3, 3)}ms')
        
        return result
    
    return wrap_func


def plane_to_screen(x, y, bounds, res):
    
    """
    coordinate trasformation from simulation space to screenspace
    
    """
    
    x_max, x_min, y_max, y_min = bounds
    res_x, res_y = res
    
    tol = 1e-7
    
    if abs(x_max - x_min) < tol:
        x_dsp = 0
    else:
        x_dsp = (x - x_min) / (x_max - x_min) * res_x
        
    if abs(y_max - y_min) < tol:
        y_dsp = 0
    else:
        y_dsp = (y - y_min) / (y_max - y_min) * res_y
    
    return x_dsp, y_dsp


def screen_to_plane(x_dsp, y_dsp, bounds, res):
    
    """
    coordinate trasformation from screenspace to simulation space
    
    """
    
    x_max, x_min, y_max, y_min = bounds
    res_x, res_y = res
    
    x = x_min + x_dsp * (x_max - x_min) / res_x 
    y = y_min + (res_y - y_dsp) * (y_max - y_min) / res_y
    
    return x, y






# APPLICATION ==============================================================

class LaplaceGame:
    
    """
    class to do everything O_O
    
    """
    
    def __init__(self, bounds=[2,-2,2,-2], res=[1000,500], fps=60):
        
        #plane settings
        self.bounds = bounds
        self.res    = res
        self.poles  = []
        self.zeros  = []
        
        #bode settings
        self.f_max = max( abs(self.bounds[0] + 1j * self.bounds[2]), abs(self.bounds[1] + 1j * self.bounds[3]) ) 
        self.f_min = self.f_max * 1e-4
        
        #display settings
        #self.r = min(self.res) / 45
        self.r = 12
        self.w = 5
        self.fps = fps
        
        #colors
        self.red        = (221, 23, 23)
        self.blue       = (45, 100, 245)
        self.fill_color = (211, 211, 211)
        
        
    def _add_pole(self, position):
        
        x, y = position
        self.poles.append(x + 1j * y)
        
        
    def _add_zero(self, position):
        
        x, y = position
        self.zeros.append(x + 1j * y)
        
        
    def _correct_poles(self, tol = 0.01):
        
        """
        snap poles to real / imag axis if close enough
        
        """
        
        x_max, x_min, y_max, y_min = self.bounds
        
        for i, p in enumerate(self.poles):
            
            if abs(p) < tol * abs((x_max - x_min) + 1j * (y_max - y_min)) : 
                self.poles[i] = 0
            elif abs(p.real) < tol * (x_max - x_min) : 
                self.poles[i] = 1j * p.imag
            elif abs(p.imag) < tol * (y_max - y_min) :
                self.poles[i] = p.real
                
    
    def _correct_zeros(self, tol = 0.01):
        
        """
        snap zeros to real / imag axis if close enough
        
        """
        
        x_max, x_min, y_max, y_min = self.bounds
        
        for i, z in enumerate(self.zeros):
            
            if abs(z) < tol * abs((x_max - x_min) + 1j * (y_max - y_min)) :
                self.zeros[i] = 0
            elif abs(z.real) < tol * (x_max - x_min) :
                self.zeros[i] = 1j * z.imag
            elif abs(z.imag) < tol * (y_max - y_min) :
                self.zeros[i] = z.real
                
                
                
    def _clean_up_poles_zeros(self, tol=0.001):
        
        """
        if poles and zeros overlapp, delete both
        
        """
        
        for i, p in enumerate(self.poles):
            for j, z in enumerate(self.zeros):
                
                if abs(p - z) < tol:
                    self.poles.pop(i)
                    self.zeros.pop(j)
                    
                    
        
                
                
    def _pole_is_selected(self, position):
        
        """
        check if position is within radius of pole
        return index of pole
        
        """
        
        res = self.Plane.get_size()

        x, y = screen_to_plane( *position, self.bounds, res)
        
        plane_r = screen_to_plane( self.r, 0, self.bounds, res)[0] - screen_to_plane( 0, 0, self.bounds, res)[0]
        
        ind = np.argwhere( (abs( np.array(self.poles) - (x + 1j*y)) <= plane_r) | (abs( np.array(self.poles) - (x - 1j*y)) <= plane_r) )
        
        return ind

    
    def _zero_is_selected(self, position):
        
        """
        check if position is within radius of zero
        return index op zero
        
        """
        
        res = self.Plane.get_size()

        x, y = screen_to_plane( *position, self.bounds, res)
        
        plane_r = screen_to_plane( self.r, 0, self.bounds, res)[0] - screen_to_plane( 0, 0, self.bounds, res)[0]
        
        ind = np.argwhere( (abs( np.array(self.zeros) - (x + 1j*y)) <= plane_r) | (abs( np.array(self.zeros) - (x - 1j*y)) <= plane_r) )
        
        return ind
    
                
    def _compute_H_frequency_domain(self):
        
        
        """
        compute frequency response from poles and zeros 
        
        """
        
        res_x, res_y = self.Bode_mag.get_size()
        
        self.f = np.logspace( np.log10(self.f_min), np.log10(self.f_max), res_x )
        
        iw = 2j * np.pi * self.f
        
        self.H = np.ones(self.f.size) + 1j * 0
        
        for p in np.array(self.poles):
            
            if p.imag != 0:
                self.H /= (iw - p) * (iw - p.conjugate())
            else:
                self.H /= (iw - p)
                
        for z in np.array(self.zeros):
            
            if z.imag != 0:
                self.H *= (iw - z) * (iw - z.conjugate())
            else:
                self.H *= (iw - z)
                
                
                
                
    def _update(self):
        
        """
        this function updates everything
        
        """
        
        self._correct_poles(tol = 0.01)
        self._correct_zeros(tol = 0.01)
        
        self._clean_up_poles_zeros(tol = 0.001)
        
        self._compute_H_frequency_domain()
        
                
                
    def _draw_poles(self):
        
        """
        draw poles as red crosses
        
        
        """
        
        
        red = self.red
        
        res = self.Plane.get_size()
        
        d = self.r / np.sqrt(2)
                
        for p in self.poles:
        
            x, y = p.real, abs(p.imag)
            
            if y > 0: 
                
                t1 = f"{round(x,2)}+{round(y,2)}j" if x != 0 else f"{round(y,2)}j"
                
                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp-d), (x2_dsp+d, y2_dsp+d), self.w+1)

                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp+d), (x2_dsp+d, y2_dsp-d), self.w+1)
                
                text = self.small_font.render(t1 , True, self.red)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                
                
                t2 = f"{round(x,2)}-{round(y,2)}j" if x != 0 else f"-{round(y,2)}j"
                
                x1_dsp, y1_dsp = plane_to_screen(x, -y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, -y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp-d), (x2_dsp+d, y2_dsp+d), self.w+1)

                x1_dsp, y1_dsp = plane_to_screen(x, -y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, -y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp+d), (x2_dsp+d, y2_dsp-d), self.w+1)
                
                text = self.small_font.render(t2 , True, self.red)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                
            else:
                
                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp-d), (x2_dsp+d, y2_dsp+d), self.w+1)

                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                x2_dsp, y2_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.line(self.Plane, red, (x1_dsp-d, y1_dsp+d), (x2_dsp+d, y2_dsp-d), self.w+1)
                
                text = self.small_font.render(f"{round(x,2)}" , True, self.red)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                
        
            
    def _draw_zeros(self):
        
        """
        draw zeros as blue circles
        
        """
    
        blue = self.blue
        
        res = self.Plane.get_size()
            
        for z in self.zeros:
            
            x, y = z.real, abs(z.imag)
            
            if y > 0:
                
                t1 = f"{round(x,2)}+{round(y,2)}j" if x != 0 else f"{round(y,2)}j"
                
                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.circle(self.Plane, blue, (x1_dsp, y1_dsp), self.r, self.w)  
                
                text = self.small_font.render(t1 , True, self.blue)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                
                
                t2 = f"{round(x,2)}-{round(y,2)}j" if x != 0 else f"-{round(y,2)}j"
                
                x1_dsp, y1_dsp = plane_to_screen(x, -y, self.bounds, res)
                pygame.draw.circle(self.Plane, blue, (x1_dsp, y1_dsp), self.r, self.w)  
                
                text = self.small_font.render(t2 , True, self.blue)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                
                
                
            else:
                
                
                x1_dsp, y1_dsp = plane_to_screen(x, y, self.bounds, res)
                pygame.draw.circle(self.Plane, blue, (x1_dsp, y1_dsp), self.r, self.w)
                
                text = self.small_font.render(f"{round(x,2)}" , True, self.blue)
                self.Plane.blit(pygame.transform.flip(text, False, True), (x1_dsp+7, y1_dsp+5))
                    
            
            
    def _draw_coordinate_system_plane(self):
        
        """
        draw coordinate system of complex plane
        
        """
        
        x_max, x_min, y_max, y_min = self.bounds
        
        res = self.Plane.get_size()
        
        
        #axes
        x1_dsp, y1_dsp = plane_to_screen(x_max, 0, self.bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(x_min, 0, self.bounds, res)
        pygame.draw.line(self.Plane, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        x1_dsp, y1_dsp = plane_to_screen(0, y_max, self.bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(0, y_min, self.bounds, res)
        pygame.draw.line(self.Plane, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        
        #unit_circle
        x_dsp, y_dsp = plane_to_screen(0, 0, self.bounds, res)
        r    , _     = plane_to_screen(1, 0, self.bounds, res)
        pygame.draw.circle(self.Plane, (0,0,0), (x_dsp, y_dsp), r-x_dsp , self.w)
        
        #numbers
        text = self.font.render("1" , True, (0,0,0))
        x_dsp, y_dsp = plane_to_screen(1, 0, self.bounds, res)
        self.Plane.blit(pygame.transform.flip(text, False, True), (x_dsp, y_dsp))
        
        text = self.font.render("1" , True, (0,0,0))
        x_dsp, y_dsp = plane_to_screen(0, 1, self.bounds, res)
        self.Plane.blit(pygame.transform.flip(text, False, True), (x_dsp-15, y_dsp))
        
        text = self.font.render("-1" , True, (0,0,0))
        x_dsp, y_dsp = plane_to_screen(-1, 0, self.bounds, res)
        self.Plane.blit(pygame.transform.flip(text, False, True), (x_dsp-20, y_dsp))
        
        text = self.font.render("-1" , True, (0,0,0))
        x_dsp, y_dsp = plane_to_screen(0, -1, self.bounds, res)
        self.Plane.blit(pygame.transform.flip(text, False, True), (x_dsp-20, y_dsp-20))
        
        #title
        text = self.font.render("complex plane" , True, (0,0,0))
        self.Plane.blit(pygame.transform.flip(text, False, True), (5,5))
        
        
        
    def _draw_coordinate_system_bode_mag(self):
        
        """
        draw coordinate system of magnitude in  bode plot
        
        """
        
        
        res = self.Bode_mag.get_size()
        
        H_abs_dB = 20 * np.log10(abs(self.H))
        
        if max(H_abs_dB) == min(H_abs_dB):
            H_max, H_min = 1, -1
        else:
            H_max, H_min = max(H_abs_dB), min(H_abs_dB)
            
        f_max, f_min = max(self.f)  , min(self.f)
        
        bounds = f_max, f_min, H_max, H_min
        
        #axes
        x1_dsp, y1_dsp = plane_to_screen(f_max, 0, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, 0, bounds, res)
        pygame.draw.line(self.Bode_mag, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        #x1_dsp, y1_dsp = plane_to_screen(f_max, H_max, bounds, res)
        #x2_dsp, y2_dsp = plane_to_screen(f_min, H_max, bounds, res)
        #pygame.draw.line(self.Bode_mag, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        x1_dsp, y1_dsp = plane_to_screen(f_max, H_min, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, H_min, bounds, res)
        pygame.draw.line(self.Bode_mag, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        x1_dsp, y1_dsp = plane_to_screen(f_min, H_max, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, H_min, bounds, res)
        pygame.draw.line(self.Bode_mag, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        #title
        text = self.font.render("bode magnitude" , True, (0,0,0))
        self.Bode_mag.blit(pygame.transform.flip(text, False, True), (5,5))
        
    
        
    def _draw_coordinate_system_bode_phase(self):
        
        """
        draw coordinate system of phase in  bode plot
        
        """
        
        res = self.Bode_mag.get_size()
        
        H_ph = np.unwrap(np.angle(self.H, deg=True), 180)
        
        if max(H_ph) == min(H_ph):
            H_max, H_min = 1, -1
        else:
            H_max, H_min = max(H_ph), min(H_ph)
            
        f_max, f_min = max(self.f)  , min(self.f)
        
        bounds = f_max, f_min, H_max, H_min
        
        
        #axes
        x1_dsp, y1_dsp = plane_to_screen(f_max, 0, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, 0, bounds, res)
        pygame.draw.line(self.Bode_phase, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        x1_dsp, y1_dsp = plane_to_screen(f_max, H_max, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, H_max, bounds, res)
        pygame.draw.line(self.Bode_phase, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        #x1_dsp, y1_dsp = plane_to_screen(f_max, H_min, bounds, res)
        #x2_dsp, y2_dsp = plane_to_screen(f_min, H_min, bounds, res)
        #pygame.draw.line(self.Bode_phase, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        x1_dsp, y1_dsp = plane_to_screen(f_min, H_max, bounds, res)
        x2_dsp, y2_dsp = plane_to_screen(f_min, H_min, bounds, res)
        pygame.draw.line(self.Bode_phase, (0,0,0), (x1_dsp, y1_dsp), (x2_dsp, y2_dsp), self.w)
        
        
        #title
        text = self.font.render("bode phase" , True, (0,0,0))
        self.Bode_phase.blit(pygame.transform.flip(text, False, True), (5,5))
        
        
        
    def _draw_bode_mag(self):
        
        """
        draw magnitude of frequency response H in bode plot
        
        """
        
        H_abs_dB = 20 * np.log10(abs(self.H))
        
        res = self.Bode_mag.get_size()
        
        H_max, H_min = max(H_abs_dB), min(H_abs_dB)
        f_max, f_min = max(self.f)  , min(self.f)
        
        offset = (H_max - H_min) / 20
        bounds = f_max, f_min, H_max + offset, H_min - offset


        #curve
        f_lin = np.linspace(f_min, f_max, self.H.size)
        
        Pts = [ plane_to_screen(f, H, bounds, res) for f, H in zip(f_lin, H_abs_dB) ]
        
        pygame.draw.lines(self.Bode_mag, self.red, False, Pts, self.w)
        
        #text
        text = self.small_font.render(f"{round(H_abs_dB[0],1)}dB" , True, self.red)
        x_dsp, y_dsp = plane_to_screen(f_min, H_abs_dB[0], bounds, res)
        y_offset = 5 if H_abs_dB[0] < H_max else -25
        self.Bode_mag.blit(pygame.transform.flip(text, False, True), (x_dsp+5, y_dsp+y_offset))
        
        
    def _draw_bode_phase(self):
        
        """
        draw phase of frequency response H in bode plot
        
        """
        
        H_ph = np.unwrap(np.angle(self.H, deg=True), 180)
        
        res = self.Bode_phase.get_size()
        
        H_max, H_min = max(H_ph), min(H_ph)
        f_max, f_min = max(self.f)  , min(self.f)
        
        offset = (H_max - H_min) / 20
        bounds = f_max, f_min, H_max + offset, H_min - offset
        
        
        #curve
        f_lin = np.linspace(f_min, f_max, self.H.size)
        
        Pts = [ plane_to_screen(f, H, bounds, res) for f, H in zip(f_lin, H_ph) ]
        
        pygame.draw.lines(self.Bode_phase, self.red, False, Pts, self.w)
        
        
        #text
        text = self.small_font.render(f"{round(H_ph[0],1)}°" , True, self.red)
        x_dsp, y_dsp = plane_to_screen(f_min, H_ph[0], bounds, res)
        y_offset = 5 if H_ph[0] < H_max else -25
        self.Bode_phase.blit(pygame.transform.flip(text, False, True), (x_dsp+5, y_dsp+y_offset))
        
        text = self.small_font.render(f"{round(H_ph[-1],1)}°" , True, self.red)
        x_dsp, y_dsp = plane_to_screen(f_max, H_ph[-1], bounds, res)
        y_offset = 5 if H_ph[-1] < H_max else -25
        self.Bode_phase.blit(pygame.transform.flip(text, False, True), (x_dsp-50, y_dsp+y_offset))
        
        
        
    def _render(self):
        
        """
        call every _draw function
        
        """
        
        res_x, res_y = self.res
        res_min = min(res_x, res_y)
        
        
        #laplace plane
        self.Plane.fill(self.fill_color)
        self._draw_coordinate_system_plane()
        self._draw_zeros()
        self._draw_poles()
        self.Dis.blit(pygame.transform.flip(self.Plane, False, True), (0,0))
        
        
        #bode plot mag
        self.Bode_mag.fill(self.fill_color)
        self._draw_bode_mag()
        self._draw_coordinate_system_bode_mag()
        self.Dis.blit(pygame.transform.flip(self.Bode_mag, False, True), (res_min,0))
        
        
        # bode plot phase
        self.Bode_phase.fill(self.fill_color)
        self._draw_bode_phase()
        self._draw_coordinate_system_bode_phase()
        self.Dis.blit(pygame.transform.flip(self.Bode_phase, False, True), (res_min,res_y//2))
        
        
        #update screen
        pygame.display.update()
        
        
        
    def _resize(self):
        
        """
        handle resizing of main window
        
        """
        
        #save new size
        self.res = self.Dis.get_size()
        
        #get new size back
        res_x, res_y = self.res
        res_min = min(res_x, res_y)
        
        #update surfaces
        self.Plane       = pygame.Surface((res_min, res_min))
        self.Bode_mag    = pygame.Surface((res_x-res_min, res_y//2))
        self.Bode_phase  = pygame.Surface((res_x-res_min, res_y//2))
        
        
    def _setup(self):
        
        """
        setup everything, especialy the pygame variables
        
        """

        #init pygame
        pygame.init()

        #init fonts
        pygame.font.init()
        font_path = r"fonts"
        self.font = pygame.font.Font( os.path.join(font_path, "OpenSans-Bold.ttf"), 18)
        self.small_font = pygame.font.Font( os.path.join(font_path, "OpenSans-Bold.ttf"), 14)
        
        #init display
        self.Dis = pygame.display.set_mode(self.res, pygame.RESIZABLE)
        pygame.display.set_caption("Interactive Bode Plot")
        
        #add surfaces
        res_x, res_y = self.res
        
        res_min = min(res_x, res_y)
        #res_max = max(res_x, res_y)
        
        self.Plane       = pygame.Surface((res_min, res_min))
        self.Bode_mag    = pygame.Surface((res_x-res_min, res_y//2))
        self.Bode_phase  = pygame.Surface((res_x-res_min, res_y//2))
        
        #init clock
        self.clock = pygame.time.Clock()
        
        #loop termination condition
        self.active = True
                
        
                
    def _end(self):
        
        """
        quit the pygame variables
        
        """
        
        pygame.font.quit()
        pygame.quit()
                
            
    def run(self):
        
        """
        function that runs the App, includes the main loop
        
        """
        
        
        self._setup()
        
        #condition for updating screen
        update_needed = True
        
        
        while self.active:
            
            #continuous mouse events
            leftclick, middleclick, rightclick = pygame.mouse.get_pressed()
            
            #some positions and dimensions
            position = pygame.mouse.get_pos()
            x_dsp, y_dsp = position
            
            res_plane = self.Plane.get_size()
            res_x_plane, res_y_plane = res_plane
            
            res_bode_mag = self.Bode_mag.get_size()
            res_x_bode_mag, res_y_bode_mag = res_bode_mag
            
            res_bode_phase = self.Bode_mag.get_size()
            res_x_bode_phase, res_y_bode_phase = res_bode_phase
            
            
            #check selections
            ind_pole = self._pole_is_selected(position)
            ind_zero = self._zero_is_selected(position)
            
            
            #check events
            for event in pygame.event.get():
                
                #quit
                if event.type == pygame.QUIT:
                    self.active = False
                    
                    
                #resize event
                if event.type == pygame.VIDEORESIZE:
                    update_needed = True
                    
                    if event.w / event.h < 2:
                        self.Dis = pygame.display.set_mode(self.res, pygame.RESIZABLE)
                    else:
                        self._resize()
                
                    
                #key events
                if event.type == pygame.KEYDOWN:
                    update_needed = True
                    
                    if event.key == pygame.K_ESCAPE:
                        self.active = False
                        
                    if event.key == pygame.K_r:
                        self.poles = []
                        self.zeros = []
                        
                        
                        
                        
                #mouse events
                if event.type == pygame.MOUSEBUTTONDOWN and ind_pole.size + ind_zero.size == 0:
                    update_needed = True
                    
                    if x_dsp < res_x_plane:
                        
                        #left mouse button    
                        if event.button == 1:
                            x, y = screen_to_plane( *position, self.bounds, res_plane)
                            self._add_pole((x, y))
                            

                        #right button    
                        if event.button == 3:
                            x, y = screen_to_plane( *position, self.bounds, res_plane)
                            self._add_zero((x, y))
                            
                            
                            
            
                            
            #drag poles
            if ind_pole.size > 0 and leftclick and x_dsp < res_x_plane:
                x, y = screen_to_plane(*position, self.bounds, res_plane)
                self.poles[int(min(ind_pole))] = x + 1j * y
                
                update_needed = True
                
            #drag zeros
            if ind_zero.size > 0 and leftclick and x_dsp < res_x_plane:
                x, y = screen_to_plane(*position, self.bounds, res_plane)
                self.zeros[int(min(ind_zero))] = x + 1j * y              
                
                update_needed = True
                
                            
                            
            
            
            
            if update_needed:
            
                #update backend, data etc.
                self._update()
                
            
                #draw everything
                self._render()

        
            #limit fps
            self.clock.tick(self.fps)
            
            update_needed = False
        
        
        self._end()
        

# MAIN ===================================================================

def main():
    
    L = LaplaceGame(res=(900,300), fps=90)
    L._add_pole((-0.5, 0))
    L._add_zero((0.75, 1.5))
    L.run()
    
    
if __name__ == '__main__':
    main()



