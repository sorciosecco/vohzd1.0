
import os

maindir = os.path.dirname(os.path.dirname(__file__))

# enviroment variables
os.environ["DIR_INSTALL"] = maindir

# These will be set during the first call of main function
ACTIVITY=None
BACKFEEL=False
BALANCE=False
FIT=False
HIGHTHRESHOLD=None
LATENT=None
LOWTHRESHOLD=None
GRIDSEARCH=False
METHOD=None
MODEL=''
MULTICLASS=False
NPARA=False
NUMBER=None
PERCENTAGE=None
PREDICT=False
PROBACUTOFF=None
SAVEMODEL=False
SAVEPRED=False
SEED=666
STRATEGY=False
VERBOSE=0

N=0
NAMES=None
workdir=''
VAR_NAMES=None
