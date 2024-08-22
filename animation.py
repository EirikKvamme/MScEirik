import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display
from matplotlib.colors import Normalize
from tqdm import tqdm

def animation_local(data, name, unit,depth,vmin,vmax,cmap):
    if cmap:
        cmap = cmap
    else:
        cmap = 'viridis'

    T = len(data)
    if vmin and vmax:
        vmin, vmax = vmin, vmax
    else:
        vmin = data.min().values  # Minimum value for color scaling
        vmax = data.max().values  # Maximum value for color scaling

    norm = Normalize(vmin=vmin, vmax=vmax)  # Ensure linear normalization

    # Adjust the figure size and layout spacing
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)  # Adjust these as necessary

    ax1.set_xticks(np.linspace(data.X.min(), data.X.max(), num=5))  
    ax1.set_yticks(np.linspace(data.Y.min(), data.Y.max(), num=5))  
    ax1.set_ylabel('Latitude [$^o$N]')
    ax1.set_xlabel('Longitude [$^o$E]')

    cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.1,colors='black')
    ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)

    mesh = ax1.pcolor(data.X, data.Y, data[0][0], cmap=cmap, norm=norm)
    
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Positioning the colorbar
    cbar = fig.colorbar(mesh, cax=cbar_ax, ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label(unit, labelpad=20)  # Adjust labelpad as necessary

    pbar = tqdm(total=T, desc="Generating Frames")

    def update_plot(frame):
        ax1.clear()
        ax1.set_title(f'{name} at t={data.time[int(frame)].values}')
        ax1.set_xlabel('Longitude [$^o$E]')
        ax1.set_ylabel('Latitude [$^o$N]')
        cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.1,colors='black')
        ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)
        ax1.pcolor(data.X, data.Y, data[frame][0], cmap=cmap, norm=norm)
        pbar.update(1)

    ani = FuncAnimation(fig, update_plot, frames=T, interval=100)
    ani.save(f'{name}.gif', writer='pillow', progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    display(Image(f'{name}.gif'))


def animation_local_3D(data, name, unit,depth,vmin,vmax,cmap):
    if cmap:
        cmap = cmap
    else:
        cmap = 'viridis'

    T = len(data)
    if vmin and vmax:
        vmin, vmax = vmin, vmax
    else:
        vmin = data.min().values  # Minimum value for color scaling
        vmax = data.max().values  # Maximum value for color scaling

    norm = Normalize(vmin=vmin, vmax=vmax)  # Ensure linear normalization

    # Adjust the figure size and layout spacing
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)  # Adjust these as necessary

    ax1.set_xticks(np.linspace(data.X.min(), data.X.max(), num=5))  
    ax1.set_yticks(np.linspace(data.Y.min(), data.Y.max(), num=5))  
    ax1.set_ylabel('Latitude [$^o$N]')
    ax1.set_xlabel('Longitude [$^o$E]')

    cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.2,colors='grey')
    ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)

    mesh = ax1.pcolor(data.X, data.Y, data[0], cmap=cmap, norm=norm)
    
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Positioning the colorbar
    cbar = fig.colorbar(mesh, cax=cbar_ax, ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label(unit, labelpad=20)  # Adjust labelpad as necessary

    pbar = tqdm(total=T, desc="Generating Frames")

    def update_plot(frame):
        ax1.clear()
        ax1.set_title(f'{name} at t={data.time[int(frame)].values}')
        ax1.set_xlabel('Longitude [$^o$E]')
        ax1.set_ylabel('Latitude [$^o$N]')
        cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.2,colors='grey')
        ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)
        ax1.pcolor(data.X, data.Y, data[frame], cmap=cmap, norm=norm)
        pbar.update(1)

    ani = FuncAnimation(fig, update_plot, frames=T, interval=100)
    ani.save(f'{name}.gif', writer='pillow', progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    display(Image(f'{name}.gif'))