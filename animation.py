import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from IPython.display import Image, display
from matplotlib.colors import Normalize
from tqdm import tqdm
import xarray as xr
from IPython.display import Video
# from PIL import Image


def animation_local_gif(data=xr.DataArray, name=str(), unit=str(),depth=xr.DataArray,vmin=None,vmax=None,cmap=None,interval=200):
    """
    Data cannot contain Z coords!
    """

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
        cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.1,colors='black')
        ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)
        ax1.pcolor(data.X, data.Y, data[frame], cmap=cmap, norm=norm)
        pbar.update(1)

    ani = FuncAnimation(fig, update_plot, frames=T, interval=interval)
    ani.save(f'{name}.gif', writer='pillow', progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)


def animation_local_mp4(data=xr.DataArray, name=str(), unit=str(),depth=xr.DataArray,vmin=None,vmax=None,cmap=None):
    """
    Data cannot contain Z coords!
    """

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
        cntr = ax1.contour(depth.X,depth.Y,depth,alpha=0.1,colors='black')
        ax1.clabel(cntr, fmt = "%2.1f",use_clabeltext=True)
        ax1.pcolor(data.X, data.Y, data[frame], cmap=cmap, norm=norm)
        pbar.update(1)

    ani = FuncAnimation(fig, update_plot, frames=T, interval=200)
    # Define the FFMpegWriter
    writer = FFMpegFileWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f'{name}.mp4', writer=writer, progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    display(Video(f'{name}.mp4'))


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


def animate_eddies(eddies_full,hor_vel,fname):
    # 1 = warm, 2 = cold, 3 = current, 4 = stream
    import matplotlib.colors as mcolors
    name = 'Eddies'
    # Define the custom colormap
    colors = ['red', 'blue', 'yellow', 'green']
    cmap = mcolors.ListedColormap(colors)

    # Define the normalization
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(15,4),layout='constrained')
    ax.set_title(f'{name} at t={eddies_full.time[0].values}')
    ax.set_xlabel('Longitude [$^o$E]')
    ax.set_ylabel('Latitude [$^o$N]')
    mesh = ax.pcolormesh(hor_vel.X,hor_vel.Y,hor_vel[0],vmin=0,vmax=1.2,cmap='binary_r')
    cbar = fig.colorbar(mesh)
    cbar.set_label('Horizontal velocity magnitude [ms$^{-1}$]')
    ax.pcolormesh(eddies_full.X,eddies_full.Y,eddies_full[0],cmap=cmap, norm=norm)
    x = [-21,1,1,-21,-21]
    y = [71,71,74,74,71]
    ax.plot(x,y,color='orange',linestyle='--', label='Area off eddy centerpoint detection')
    ax.legend()

    T = len(eddies_full.time)
    pbar = tqdm(total=T, desc="Generating Frames")
    def update_plot(frame):
        ax.clear()
        ax.set_title(f'{name} at t={eddies_full.time[int(frame)].values}')
        ax.set_xlabel('Longitude [$^o$E]')
        ax.set_ylabel('Latitude [$^o$N]')
        ax.pcolormesh(hor_vel.X,hor_vel.Y,hor_vel[frame],cmap='binary_r')
        ax.pcolormesh(eddies_full.X, eddies_full.Y, eddies_full[frame], cmap=cmap, norm=norm)
        x = [-21,1,1,-21,-21]
        y = [71,71,74,74,71]
        ax.plot(x,y,color='orange',linestyle='--', label='Area off eddy centerpoint detection')
        ax.legend()
        pbar.update(1)

    ani = FuncAnimation(fig, update_plot, frames=T, interval=300)
    ani.save(f'{fname}.gif', writer='pillow', progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    display(Image(f'{fname}.gif'))