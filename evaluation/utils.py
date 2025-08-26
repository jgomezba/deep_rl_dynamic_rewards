from datetime import datetime
import imageio
from pathlib import Path


def save_gif(frames, episode_num):
    """
    Save a list of frames as a GIF with timestamped filename.

    Args:
        frames (list of np.ndarray): Frames captured during the episode.
        episode_num (int): Episode number for filename.
    """
    gifs_dir = Path("gifs")
    gifs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = gifs_dir / f"episode_{episode_num}_{timestamp}.gif"
    imageio.mimsave(gif_path, frames, duration=0.25)
    print(f"GIF saved at {gif_path}")