# This is a script for automating the fitting of multiple datasets at once
set -e
# ARGUMENT PARSING START #
PROMPTING=false
BUILD=true

if [ $# -eq 0 ] ; then
  echo "This is a script for the automatic fitting of a model"
  echo "This will automatically write configurations into a folder and execute them"
  PROMPTING=true
fi

if [ "$1" == "-h" ] ; then
  echo "Usage: just-run-it.sh [-h | -f | CONFIG_DIRECTORY] [DATA_FILE MODEL_FILE GUESS_FILE FUNCTIONS_FILE]"
  echo ""
  echo "-h               | Shows this help"
  echo "-f               | Performs automated fitting/config building"
  echo ""
  echo "Positional Arguments: "
  echo ""
  echo "CONFIG DIRECTORY | Exisitng configuration directory (provided in lieu of -f)"
  echo "DATA FILE        | Path to file containing the data to wrangle"
  echo "MODEL FILE       | Path to file containing the model"
  echo "GUESS FILE       | Path to file containing initial parameter guess"
  echo "FUNCTIONS FILE   | Path to file containing data ingestion functions"
  echo "OUTPUT_DIRECTORY | Path to the directory to hold the fitting results"
  echo ""
  echo "Report bugs to dwu402@aucklanduni.ac.nz"
  exit 0
fi

if $PROMPTING ; then
  echo "Do you already have a configurations folder ready to go?"
  DIRINPUT=$(read -p "If so, please enter it here:")
  if [ -z "DIRINPUT" ] ;  then
    DIR="configs/build"
  else
    DIR=$DIRINPUT
    BUILD=false
  fi
elif [ -z "$1" ] || [ $1 == "-f" ]; then
  DIR="configs/build"
else
  DIR=$1
  BUILD=false
fi

if $BUILD ; then
  if $PROMPTING || [ -z "$2" ] ; then
    DATAPATH=$(read -p "Please enter the path the data file: ")
  else
    DATAPATH=$2
  fi

  if $PROMPTING || [ -z "$3" ] ; then
    MODELPATH=$(read -p "Please enter the path to the model file: ")
  else
    MODELPATH=$3
  fi

  if $PROMPTING || [ -z "$4" ] ; then
    PARAMPATH=$(read -p "Please enter the path to the initial parameter guess file: ")
  else
    PARAMPATH=$4
  fi

  if $PROMPTING || [ -z "$5" ] ; then
    FUNCTIONS=$(read -p "Please enter the path to the file containing the data ingestion functions: ")
  else
    FUNCTIONS=$5
  fi

  if $PROMPTING ; then
    OUTPUTDIR=$(read -p "Please enter the path to the outputs directory: ")
  elif [ -z "$6" ] ; then
      OUTPUTDIR="outputs"
  else
    OUTPUTDIR=$6
  fi
# ARGUMENT PARSING END #

# WRANGLING START #
  python data/wrangle_data.py -f "$DATAPATH"
  DATADIR=$(dirname "$DATAPATH")

  if [ ! -d "$DIR" ] ; then
    mkdir $DIR
  fi

  ICS="0 0 0" # this value does not matter, overwritten by data
  for datafile in $DATADIR/*.csv; do
    datafilebase=$(echo "$datafile" | awk -F "/" '{print $NF}')
    configpath="$DIR/${datafilebase%.*}.config"
    echo "mf $MODELPATH
pf $PARAMPATH
df $datafile
ts $ICS
iv $ICS
pd $FUNCTIONS treat_data
ef $FUNCTIONS error_fn
dv $FUNCTIONS data_plot" > "$configpath"
  done
fi
# WRANGLING END #

# FITTING START #
if [ ! -d "$OUTPUTDIR" ] ; then
  mkdir "$OUTPUTDIR"
fi

for config in $DIR/*.config; do
  filebase=$(echo "$config" | awk -F "/" '{print $NF}')
  outputfile="$OUTPUTDIR/${filebase%.*}.out"
  python main.py -a f -c $config -o $outputfile
done
# FITTING END #

exit 0
