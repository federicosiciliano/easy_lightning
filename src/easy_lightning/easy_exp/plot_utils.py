import matplotlib.pyplot as plt
import numpy as np

# other_color_combinations = {
#     1: ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666'],
#     2: ['#1b9e77', '#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'],
#     3: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00'],
#     4: ['#e41a1c','#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
#     5: ['#66c2a5','#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
#     6: ['#8dd3c7','#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
# }

# IBM Color Blind Safe Palette (originally includes black and white)
color_blind_safe = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"] #removed white and black
#Paul Tol palettes
paul_tol_bright = {'blue': '#4477AA', 'red': '#EE6677', 'green': '#228833', 'yellow': '#CCBB44', 'cyan': '#66CCEE', 'purple': '#AA3377', 'grey': '#BBBBBB'} 
paul_tol_high_contrast = {'blue': '#004488', 'yellow': '#DDAA33', 'red': '#BB5566'} #removed white and black
paul_tol_vibrant = {'orange': '#EE7733', 'blue': '#0077BB', 'cyan': '#33BBEE', 'magenta': '#EE3377', 'red': '#CC3311', 'teal': '#009988', 'grey': '#BBBBBB'}
paul_tol_muted = {'rose': '#CC6677', 'indigo': '#332288', 'sand': '#DDCC77', 'green': '#117733', 'cyan': '#88CCEE', 'wine': '#882255', 'teal': '#44AA99', 'olive': '#999933', 'purple': '#AA4499', 'grey': '#DDDDDD'}
paul_tol_medium_contrast = {'light_blue': '#6699CC', 'dark_blue': '#004488', 'light_yellow': '#EECC66', 'dark_red': '#994455', 'dark_yellow': '#997700', 'light_red': '#EE99AA'}
paul_tol_pale = {'pale blue': '#BBCCEE', 'pale cyan': '#CCEEFF', 'pale green': '#CCDDAA', 'pale yellow': '#EEEEBB', 'pale red': '#FFCCCC', 'pale grey': '#DDDDDD'}
paul_tol_dark = {'dark blue': '#222255', 'dark cyan': '#225555', 'dark green': '#225522', 'dark yellow': '#666633', 'dark red': '#663333', 'dark grey': '#555555'}
paul_tol_bright = {'light_blue': '#77AADD', 'orange': '#EE8866', 'light_yellow': '#EEDD88', 'pink': '#FFAABB', 'light_cyan': '#99DDFF', 'mint': '#44BB99', 'pear': '#BBCC33', 'olive': '#AAAA00', 'pale_grey': '#DDDDDD'}

# https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
contrastive_colors = [
    ["#FFC20A", "#0C7BDC"],
    ["#994F00", "#006CD1"],
    ["#E1BE6A", "#40B0A6"],
    ["#E66100", "#5D3A9B"],
    ["#1AFF1A", "#4B0092"],
    ["#FEFE62", "#D35FB7"],
    ["#005AB5", "#DC3220"],
    ["#1A85FF", "#D41159"]
]

all_color_combinations = {
    'color_blind_safe': color_blind_safe,
    'paul_tol_bright': paul_tol_bright,
    'paul_tol_high_contrast': paul_tol_high_contrast,
    'paul_tol_vibrant': paul_tol_vibrant,
    'paul_tol_muted': paul_tol_muted,
    'paul_tol_medium_contrast': paul_tol_medium_contrast,
    'paul_tol_pale': paul_tol_pale,
    'paul_tol_dark': paul_tol_dark,
    'paul_tol_bright': paul_tol_bright,
    'contrastive_colors': contrastive_colors
}

number_of_colors_to_combination = {len(colors): name for name, colors in all_color_combinations.items()}

def get_plot_colors(num_colors:int=2, seed:int=42, color_combination:str=None):
    if color_combination is not None:
        color_combination = all_color_combinations[color_combination]
    else:
        possible_color_combinations = [name for name, colors in all_color_combinations.items() if len(colors) >= num_colors]
        np.random.seed(seed)
        color_combination = all_color_combinations[np.random.choice(possible_color_combinations)]
    np.random.seed(seed)
    return np.random.choice(color_combination, num_colors, replace=False)
    
def obtain_markers(number_of_markers:int=2, seed:int=42):
    list_of_markers = ['o', 'v', 's', 'p', 'P', '*', 'X', '+', 'D', 'x', 'd']
    np.random.seed(seed)
    return np.random.choice(list_of_markers, number_of_markers, replace=False)

def attach_grid(input, line_style=None, color=None, seed:int=42):
    linestyles = ['-', '--', '-.']
    linewidths = np.linspace(0.5, 2, 20).tolist()
    if line_style is None:
        if color is None:
            np.random.seed(seed)
            return input.grid(True, linestyle=np.random.choice(linestyles),
                          linewidth=np.random.choice(linewidths))
        else:
            np.random.seed(seed)
            return input.grid(True, linestyle=np.random.choice(linestyles),
                          linewidth=np.random.choice(linewidths), color=color)