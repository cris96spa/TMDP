MLFLOW_URI = 'http://192.168.0.182:12832'

SEEDS = [2999, 58864, 29859, 25025, 23110, 60779, 8649, 51036, 31886, 12715, 16404, 17710,
17565, 18880, 21395, 46619, 9038, 20400, 59667, 60371, 2241, 22997, 54024, 13390,
59161, 5783, 27666, 29823, 2243, 36837, 59216, 65465, 31734, 57679, 20482, 53408,
32494, 11826, 24216, 49352, 19646, 12647, 47723, 10301, 12953, 52161, 30038, 33952,
14301, 10575, 45046, 56659, 59270, 9587, 62780, 60815, 27210, 13344, 23925, 63572,
33651, 57131, 42892, 60316, 50724, 43373, 51824, 57845, 58878, 22835, 41010, 4543,
37685, 52972, 65066, 21818, 22241, 21581, 4571, 43347, 39093, 53766, 20802, 4079,
30993, 59689, 17476, 34969, 9642, 64190, 21337, 38756, 64468, 33700, 7811, 43416,
46938, 63389, 13312, 54252, 11449, 49564, 2105, 15902, 16197, 10762, 12354, 4496,
15598, 21711, 38202, 32842, 30591, 52769, 39347, 40294, 48163, 47574, 18297, 38969,
34946, 21473, 27883, 63139, 37793, 52122, 19131, 9533, 24893, 62199, 9931, 18534,
18053, 64445, 29419, 52028, 65372, 19292, 20568, 50162, 48055, 41753, 12260, 51513,
57139, 14087, 31870, 52263, 12779, 35471, 38272, 55379, 25814, 34891, 54253, 10963,
19428, 4557, 28132, 47058, 34849, 17417, 20104, 62198, 48096, 54301, 5808, 34491,
23781, 44736, 18976, 55701, 52729, 4673, 30756, 32814, 53731, 23438, 50706, 60545,
5311, 31595, 6278, 29786, 30120, 57297, 24259, 46744, 17263, 28933, 15406, 17107,
1122, 50437, 40449, 21858, 53823, 22735, 65109, 45360]


LABEL_DICT = {
    'CurrMPI': 'TMPI',
    'CurrPMPO': 'D-TPPO',
    'CurrPPO': 'S-TPPO',
    'CurrQ': 'S-TQ',
    'PPO': 'PPO',
    'Q': 'Q',
    'Optimal': 'Optimal',
    '2': 'Test'
}

LINE_STYLES= ['solid']#, 'dashed', 'dashdot', 'dotted', '-.', ':']

COLORS = ['darkorange', 'green', 'red', 'c', 'blue', 'purple', 
          'brown', 'pink', 'gray', 'olive', 'cyan', 
          'lime', 'teal', 'lavender', 'tan', 'salmon', 
          'gold', 'indigo', 'maroon', 'navy', 'peru', 
          'plum', 'sienna', 'tomato', 'violet', 'wheat', 
          'yellow', 'azure', 'beige', 'coral', 'crimson', 
          'fuchsia', 'khaki', 'magenta', 
          'olive', 'orchid', 'silver', 'snow', 'tan', 
          'thistle', 'turquoise', 'yellow', 'aqua', 'black', 
           'brown', 'chartreuse', 'chocolate', 'coral', 
          'cornflower', 'darkblue', 'darkcyan', 'darkgoldenrod', 
          'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 
          'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 
          'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 
          'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 
          'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 
          'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'honeydew', 'hotpink', 
          'indianred', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 
          'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 
          'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 
          'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 
          'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 
          'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 
          'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 
          'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowh']
MARKERS = ['o', '^', 'D', 'x', 'H','+', '.', ',',  'v',  '<', '>', 
           '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 
           'd', '|', '_', 'P', 'X', 
           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


MARKERS_DICT = {
    'CurrMPI': 's', 
    'CurrPMPO': 'D', 
    'CurrPPO': '^', 
    'CurrQ': '*', 
    'PPO': 'x', 
    'Q': 'o',
    'Optimal': 'None',
    '2': 'None'
}
LINE_STYLES_DICT = {
    'CurrMPI': 'dotted', 
    'CurrPMPO': 'solid', 
    'CurrPPO': 'dashdot', 
    'CurrQ': '--', 
    'PPO': 'dashed', 
    'Q': '-.',
    'Optimal': ':',
    "2": 'solid'
}

COLORS_DICT = { # Category 10
    'CurrMPI': '#1F77B4', 
    'CurrPMPO': '#FF7F0E',  
    'CurrPPO': '#2CA02C',  
    'CurrQ': '#D62728',    
    'PPO': '#9467BD',      
    'Q': '#8C564B',     
    'Optimal': '#E377C2',
    '2': 'blue'       
}

COLORS_DICT_v2 = { # Category 10 edited
    'CurrMPI': '#1F77B4', 
    'CurrPMPO': 'darkorange',  
    'CurrPPO': 'blue',  
    'CurrQ': '#D62728',    
    'PPO': '#9467BD',      
    'Q': 'darkblue',     
    'Optimal': '#E377C2'      
}

MARKER_SIZE_DICT = {
    'CurrMPI': 6, 
    'CurrPMPO': 6, 
    'CurrPPO': 7, 
    'CurrQ': 8, 
    'PPO': 6, 
    'Q': 5,
    'Optimal': 6,
    '2': 6   
}

MARKER_FREQUENCY_DICT = {
    'CurrMPI': 1300, 
    'CurrPMPO': 2000, 
    'CurrPPO': 2400, 
    'CurrQ': 2900, 
    'PPO': 3500, 
    'Q': 3700,
    'Optimal': 4000,
    '2': 4000
}

MARKER_LOG_FREQUENCY_DICT = {
    'CurrMPI': 0.08, 
    'CurrPMPO': 0.9, 
    'CurrPPO': 0.21, 
    'CurrQ': 0.11, 
    'PPO': 0.17, 
    'Q': 0.171,
    'Optimal': 0.68,
    '2': 0.68
}



COLORS_DICT_DARK = { # COLOR BLIND FRIENDLY
    'CurrMPI': '#000000',  
    'CurrPMPO': '#E69F00',  
    'CurrPPO': '#56B4E9',  
    'CurrQ': '#D55E00',    
    'PPO': '#0072B2',      
    'Q': '#009E73',        
}


COLORS_DICT_OKABE_ITO = {
    'CurrMPI': '#E69F00',  # Orange
    'CurrPMPO': '#56B4E9',  # Sky Blue
    'CurrPPO': '#009E73',  # Bluish Green
    'CurrQ': '#0072B2',    # Blue (shifted)
    'PPO': '#D55E00',      # Vermillion (shifted)
    'Q': '#CC79A7',        # Reddish Purple (shifted from next color in sequence)
}

# Tol Bright Palette
COLORS_DICT_TOL_BRIGHT = {
    'CurrMPI': '#4477AA',  # Blue
    'CurrPMPO': '#66CCEE',  # Cyan
    'CurrPPO': '#228833',  # Green
    'CurrQ': '#CCBB44',    # Yellow
    'PPO': '#EE6677',      # Red
    'Q': '#AA3377',        # Purple
}

# Tol Muted Palette
COLORS_DICT_TOL_MUTED = {
    'CurrMPI': '#88CCEE',  # Light Blue
    'CurrPMPO': '#CC6677',  # Light Red
    'CurrPPO': '#DDCC77',  # Light Yellow
    'CurrQ': '#117733',    # Dark Green
    'PPO': '#332288',      # Dark Blue
    'Q': '#AA4499',        # Purple
}

# Tol Light Palette
COLORS_DICT_TOL_LIGHT = {
    'CurrMPI': '#77AADD',  # Light Blue
    'CurrPMPO': '#EE8866',  # Light Red
    'CurrPPO': '#EEDD88',  # Light Yellow
    'CurrQ': '#FFAABB',    # Pink
    'PPO': '#99DDFF',      # Light Cyan
    'Q': '#44BB99',        # Light Green
}