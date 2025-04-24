import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
import pandas as pd
import seaborn as sns
import h5py
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# We'll need to do things like calculate the scalar order parameter $S$ at every point and time.
# So it would probably be more efficient to just write a class that will do all these things.

class DirectorField:
    def __init__(self, filename, size):
        self.name = Path(filename).stem
        with h5py.File(filename, 'r') as f:
            self.n = f['director'][:]
            self.timepoints = self.n.shape[0]
            self.length = self.n.shape[2]
            self.width = self.n.shape[3]

            # Now we will pre-load the theta and S arrays.  These are big computations but it's more efficient up here than doing it in each function.
            self.theta = self.theta()
            self.S = self.S(size)

    def theta(self):
        logger.debug('Computing theta')
        return np.arctan2(self.n[:, 1, :, :], self.n[:, 0, :, :])

    def S(self, size):
        # This function wil return the scalar order parameter FIELD as an array
        # size is the size of coarse graining (i.e., how many neighbors we average over)

        logger.debug('Computing S')
        filtersize = 2 * size + 1

        # The uniform_filter() function does the coarse graining for us!  We use it on both theta and S
        coarse_grained_theta = uniform_filter(self.theta, size=filtersize)
        rawS = (3 * np.cos(self.theta - coarse_grained_theta) ** 2 - 1) / 2

        return uniform_filter(rawS, size=filtersize)        

    def charge(self, t, x, y, r):
        # Given a defect candidate at the point (x, y) at time t, this will compute the topological charge of this candidate

        # We will take a square contour loop with corners at (x + r, y + r), (x - r, y + r), (x - r, y - r), and (x + r, y - r)
        contour = []

        # Top contour from (x + r, y + r) to (x - r, y + r)
        for i in range(x + r, x - r, -1):                       # r better be positive or this won't work
            contour.append((i, y + r))
        
        # Left contour from (x - r, y + r) to (x - r, y - r)
        for i in range(y + r, y - r, -1):
            contour.append((x - r, i))

        # Bottom contour from (x - r, y - r) to (x + r, y - r)
        for i in range(x - r, x + r):
            contour.append((i, y - r))
        
        # Right contour from (x + r, y - r) to (x + r, y + r)
        for i in range(y - r, y + r):
            contour.append((x + r, i))
        
        contour.append(((x + r, y + r)))        # Add the initial point at the end for convenience

        # Then, we compute the director field and associated angle ALONG THE CONTOUR
        angles = [self.theta[t, a, b] for (a, b) in contour]
        differences = np.unwrap([angles[i + 1] - angles[i] for i in range(len(angles) - 1)])

        # We compute the (raw) topological charge by approximating the integral as a sum of these differences
        q = np.sum(differences) / (2 * np.pi)

        # Then, we round to the nearest half integer:
        return round(2 * q) / 2

    def find_defects(self, t, cutoff, r):
        logger.debug(f'Finding defects at time {t}')

        x_cand, y_cand = np.where(self.S[t] < cutoff)           # Filter out all the points where S < cutoff
        # Keep the ones in range to make a contour around it, and keep the ones with nonzero topological charge
        return [(x, y) for (x, y) in zip(x_cand, y_cand) if r <= x < self.length - r and r <= y < self.width - r and self.charge(t, x, y, r) != 0]

    def find_all_defects(self, cutoff, r):
        defects = {}        # We'll initialize this as an empty dictionary.  The keys will be the times.

        for t in range(self.timepoints):
            defects[t] = self.find_defects(t, cutoff, r)
            
        return defects
    
    def plot_defects(self, t, cutoff, r):
        # This function is more of a sanity check than anything.  We plot the locations of the defects against a heatmap of S.
        
        defects = self.find_defects(t, cutoff, r)

        if not defects:
            logger.warning(f'No defects found at time {t}')
            return None, (None, None)

        defects_x, defects_y = zip(*defects)

        fig, (ax_D, ax_S) = plt.subplots(1, 2, figsize=(15, 6))
        ax_D.scatter(defects_x, defects_y, s=2)                                         # Scatter plot of defects
        sns.heatmap(np.transpose(self.S[t]), ax=ax_S, cmap='RdYlBu', center=cutoff)     # Heatmap of S

        ax_S.invert_yaxis()                                         # So that 0 shows up on the bottom and not the top

        fig.suptitle(f'Topological defects and scalar order parameter at time {t} from {self.name}', fontsize=18)

        return fig, (ax_D, ax_S)

    def plot_defect_density(self, cutoff, r):
        defects = self.find_all_defects(cutoff, r)
        num_defects = [len(defects[t]) for t in range(self.timepoints)]                                             # Total number of defects at each time
        charges = [sum([self.charge(t, x, y, r) for (x, y) in defects[t]]) for t in range(self.timepoints)]         # Total charge from the defects at each time

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # The first plot will plot the number of defects by time, and the second plot is the the total charge by time
        ax1.plot(range(self.timepoints), num_defects)
        ax2.plot(range(self.timepoints), charges)

        fig.suptitle(f'Number of defects and total topological charge of all defects over time from {self.name}', fontsize=18)
        ax1.set_xlabel('Timesteps', fontsize=16)
        ax2.set_xlabel('Timesteps', fontsize=16)
        ax1.set_ylabel('Number of defects', fontsize=16)
        ax2.set_ylabel('Total topological charge', fontsize=16)

        return fig, (ax1, ax2)
    
    def correlation_length(self, nPairs):
        logger.debug('Computing correlation length')

        # We first begin by defining the correlation function -- or rather, the part that is supposed to be averaged
        def to_be_averaged(t, x1, y1, x2, y2):
            theta1 = np.arctan2(self.n[t, 1, x1, y1], self.n[t, 0, x1, y1])
            theta2 = np.arctan2(self.n[t, 1, x2, y2], self.n[t, 0, x2, y2])
            return np.cos(2 * (theta1 - theta2))

        sorted_by_distance = {}     # This is a preliminary dictionary we will use to keep account of the outputs of the to_be_averaged() based on the distance between the two points

        # The idea here is we want to average over all pairs of the same length, over all time points.  However, this is a lot of pairs.
        # So we will sample a sufficiently large number of pairs but not too large that it is computationally unmanageable.  
        # 10000 per timestep is a good number, but we will leave that up to the variable samples

        for t in range(self.timepoints):
            x1 = np.random.randint(0, self.length, size=nPairs)
            y1 = np.random.randint(0, self.width, size=nPairs)
            x2 = np.random.randint(0, self.length, size=nPairs)
            y2 = np.random.randint(0, self.width, size=nPairs)
            
            for i in range(nPairs):
                r = np.sqrt((x2[i] - x1[i]) * (x2[i] - x1[i]) + (y2[i] - y1[i]) * (y2[i] - y1[i]))
                
                if r not in sorted_by_distance:
                    sorted_by_distance[r] = [to_be_averaged(t, x1[i], y1[i], x2[i], y2[i])]
                else:
                    sorted_by_distance[r].append(to_be_averaged(t, x1[i], y1[i], x2[i], y2[i]))
        
        # Now we will take the averages
        r_data = sorted(sorted_by_distance.keys())
        C_data = [np.mean(sorted_by_distance[r]) for r in r_data]

        # We would like to curve fit this guy into the form Ae^(- r / l) + B, where A, B, and l are constants.  In this case, l is the correlation length that we want.
        def f(r, A, B, l):
            return A * np.exp(-1 * r / l) + B

        popt, pcov = curve_fit(f, r_data, C_data)       # popt = (A, B, l) for the best fit.  We really only care about l.

        return popt[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='../../../data/processeddata')     # Directory of the datasets
    parser.add_argument('--outputdir', type=str, default='../../../data/directorplots')     # Output directory of plots
    parser.add_argument('--cutoff', type=float, default=0.7)                                # The cutoff in the scalar order parameter S to identify defect candidates
    parser.add_argument('--size', type=int, default=3)                                      # Size ("radius") of coarse graining, in pixels
    parser.add_argument('--r', type=int, default=2)                                         # "Radius" of contour for topological defects
    parser.add_argument('--nPairs', type=int, default=10000)                                # Number of pairs per timestep to sample when finding correlation length

    args = parser.parse_args()

    Path(args.outputdir).mkdir(parents=True, exist_ok=True)                                 # Create output_dir if it doesn't exist already

    files = [file for file in Path(args.directory).iterdir() if file.is_file() and file.suffix == ".hdf5" and not file.name.startswith("._")]

    for file in files:
        logger.info(f'Analyzing file {file.name}')
        director = DirectorField(file, args.size)

        director.plot_defects(0, args.cutoff, args.r)[0].savefig(f'{args.outputdir}/Defects{director.name}.png')
        logger.info(f'Plots saved to {args.outputdir}/Defects{director.name}.png')

        director.plot_defect_density(args.cutoff, args.r)[0].savefig(f'{args.outputdir}/DefectDensity{director.name}.png')
        logger.info(f'Plots saved to {args.outputdir}/DefectDensity{director.name}.png')

        logger.info(f'Nematic Correlation Length: {director.correlation_length(args.nPairs)}')