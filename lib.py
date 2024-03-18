from typing import Tuple
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import numpy as np
import torch
from time import time
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


def plot_fractal(frac, roots, xlim, ylim):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(frac.T, extent=(*xlim, *ylim), cmap="viridis", interpolation="bilinear")
    # root_x = [x.real for x in roots if xlim[0] < x.real < xlim[1] and ylim[0] < x.imag < ylim[1]]
    # root_y = [x.imag for x in roots if xlim[0] < x.real < xlim[1] and ylim[0] < x.imag < ylim[1]]
    # plt.plot(root_x, root_y, ".k", ms=10)
    plt.axis("off")
    return fig
    
def make_gif(filename, update_func, duration, fps=20):
    def make_frame(t):
        fig = update_func(t / duration)
        img = mplfig_to_npimage(fig)
        plt.close(fig)
        return img
    animation = VideoClip(make_frame, duration=duration)
    animation.write_gif(filename, fps=fps)
    
class Newton:
    def __init__(self, coef_from, coef_to):
        self.coef_from = coef_from
        self.coef_to = coef_to
        # self.poly_from = Polynomial(coeffs_from)
        # self.poly_to = Polynomial(coeffs_to)
        # self.deg = self.poly_from.degree()
    
    def set_colors(self, cmap):
        self.cmap = cmap
        
    def fractal(self, p, iter=100, treshold=1e-5):                
        flag = torch.zeros((self.N, self.N), dtype=torch.bool)
        niter = torch.zeros((self.N, self.N), dtype=torch.int64)
        numbers = self.numbers.clone()
        for i in range(iter):
            tmp = numbers.clone()
            numbers = numbers - p(numbers) / p.deriv()(numbers)
            numbers[numbers.isnan()] = tmp[numbers.isnan()]
            flag[torch.abs(numbers-tmp) < treshold] = True
            niter[flag==False] += 1
            
        numbers = numbers.numpy()
        root_distance = np.array([np.abs(numbers - root) for root in p.roots()])
        groups = np.argmin(root_distance, axis=0)
        iters = niter.numpy()
        by_iter = np.array([[10*groups[i][j]+iters[i][j] for j in range(self.N)] for i in range(self.N)])
        
        return groups, by_iter
    
    def fractal_animation(self, filename, xlim, ylim, num, iter=100, treshold=1e-5, duration=5, fps=20, by_iter=False):
        self.N = num
        x_axis = np.linspace(*xlim, self.N)
        y_axis = np.linspace(*ylim, self.N)
        self.numbers = torch.empty((self.N, self.N), dtype=torch.complex128)
        coef_from, coef_to = np.array(self.coef_from), np.array(self.coef_to)
        for i in range(self.N):
            a = x_axis[i]
            for j in range(self.N):
                b = y_axis[j]
                self.numbers[i][j] = torch.complex(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
                
        def update_func(t):
            coefs = (1-t) * coef_from + t * coef_to
            poly = Polynomial(coefs)
            groups, iter_values = self.fractal(poly, iter=iter, treshold=treshold)
            if by_iter:
                fig = plot_fractal(iter_values, poly.roots(), xlim, ylim)
            else:
                fig = plot_fractal(groups, poly.roots(), xlim, ylim)
            return fig
        
        make_gif(filename, update_func, duration, fps)
        
    def fractal_zoom_animation(self, filename, xlim, ylim, point, num, iter=100, treshold=1e-5, duration=5, fps=20, by_iter=False):
        coef_from, coef_to = np.array(self.coef_from), np.array(self.coef_to)
        self.N = num
        def update_func(t):
            new_xlim = [point[0]-(1.000000001-t)*xlim[1], point[0]+(1.000000001-t)*xlim[1]]
            new_ylim = [point[1]-(1.000000001-t)*ylim[1], point[1]+(1.000000001-t)*ylim[1]]
            coefs = coef_from
            poly = Polynomial(coefs)
            x_axis = np.linspace(*new_xlim, num)
            y_axis = np.linspace(*new_ylim, num)
            self.numbers = torch.empty((num, num), dtype=torch.complex128)
            for i in range(num):
                a = x_axis[i]
                for j in range(num):
                    b = y_axis[j]
                    self.numbers[i][j] = torch.complex(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
            
            groups, iter_values = self.fractal(poly, iter=iter, treshold=treshold)
            if by_iter:
                fig = plot_fractal(iter_values, poly.roots(), new_xlim, new_ylim)
                # self.xlim = [self.xlim[0]+0.1, self.xlim[1]-0.1]
                # self.ylim = [0.90*self.ylim[0], 0.99*self.ylim[1]]
            else:
                fig = plot_fractal(groups, poly.roots(), new_xlim, new_ylim)
                # self.xlim = [0.99*self.xlim[0], 0.99*self.xlim[1]]
                # self.ylim = [0.95*self.ylim[0], 0.96*self.ylim[1]]
            return fig
        
        make_gif(filename, update_func, duration, fps)
            
        
if __name__ == '__main__':    
    # real = torch.tensor([1, -1/2, -1/2], dtype=torch.float64)
    # img = torch.tensor([0, (3**(.5))/2, -(3**(.5))/2], dtype=torch.float64)
    # roots = torch.complex(real, img)
    # cmap = [mpl.colormaps['Blues'], mpl.colormaps['Reds'], mpl.colormaps['Greens']]
    xlim = [-2.0, 2.0]
    ylim = [-2.0, 2.0]
    num = 700
    iter = 10
    treshold = 1e-5
    duration = 20
    fps = 20
    
    fractal = Newton([-1, 1, 1, 1], [-1, 1, 1, 1])
    fractal.fractal_zoom_animation("frac_5.gif", xlim, ylim, (0, 0.57),num, iter=iter, treshold=treshold, duration=duration, fps=fps, by_iter=True)
    # fractal = Newton([-1, 0, 0, 1, 0, 0, 1], [1.5, 0, 0, 1, 0, 0, 1])
    # fractal.fractal_animation("frac_4.gif", xlim, ylim, num, iter=iter, treshold=treshold, duration=duration, fps=fps, by_iter=True)
    # fractal.set_colors(cmap)
    #fractal.show(3, 3, 500, iter=10, treshold=1e-5)
        
    