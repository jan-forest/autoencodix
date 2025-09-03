#!/bin/bash
set -e

echo "=============================================================="
echo " Setting Modules"
echo "=============================================================="
module purge

MODULES="release/24.04  GCCcore/11.3.0 Python/3.10.4"
module load $MODULES
sleep 2s

echo "=============================================================="
echo " Setting Kernel"
echo "=============================================================="
# Sourcing the virtual environment
VENV_PATH="/data/horse/ws/neju219d-AUTOENCODIX/autoencodix/venv-AUTOENCODIX-capella-no-ssp"
VENV_NAME="venv-AUTOENCODIX-capella-no-ssp"
echo "Activating virtual environment: $VENV_NAME"
source $VENV_PATH/bin/activate

# Installing kernel

PYTHON_KERNEL_NAME="autoencodix-ga-kernel"
PYTHON_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$PYTHON_KERNEL_NAME"

echo "Installing kernel"
python -m ipykernel install --user --name "$PYTHON_KERNEL_NAME" --display-name="$PYTHON_KERNEL_NAME"

# Creating kerenel initialization script
echo "Creating kernel initialization script."

# Placing kernel initialization script in the kernel directory
KERNEL_INIT_SCRIPT="$PYTHON_KERNEL_DIR/kernel-init.sh"

#echo "$PYTHON_KERNEL_DIR"
#ls "$PYTHON_KERNEL_DIR"

#touch $KERNEL_INIT_SCRIPT

# Creating a kernel initialization script
echo "Creating kernel intialization script."
cat << EOF > "$KERNEL_INIT_SCRIPT"
#!/bin/bash

CONNFILE=\${1}

set -euo pipefail

echo "========================================================="
echo "Starting kernel: $PYTHON_KERNEL_NAME"

module reset

module load $MODULES

PYVENV_PATH=$VENV_PATH

source \$PYVENV_PATH/bin/activate

python -m ipykernel_launcher -f \${CONNFILE}

echo "========================================================="
# End of the script
EOF

chmod u+rwx $KERNEL_INIT_SCRIPT

# Replace the contents of kernel.json file
# echo "Replacing contents of $PYTHON_KERNEL_DIR/kernel.json"
cat << EOF > "$PYTHON_KERNEL_DIR/kernel.json"
{
  "argv": [
    "$KERNEL_INIT_SCRIPT",
    "{connection_file}"
  ],
  "display_name": "$PYTHON_KERNEL_NAME",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF

echo "Kernel installation complete."


echo "=============================================================="
echo " Creating & registering AE-ws workspace"
echo "=============================================================="
deactivate
module reset
sleep 1s

ws_allocate -F horse AE-ws 10
ws_register -F horse $HOME
ln -s $VENV_PATH $HOME/horse/neju219d-AE-ws/autoencodix/venv-gallia
#ln -s venv-AUTOENCODIX-capella-no-ssp venv-gallia


echo "=============================================================="
echo " Your next steps"
echo "=============================================================="
echo "1. Re-load the browser"
echo "2. Relocate to directory /horse/$USER-AE-ws/autoencodix/Tutorials"
echo "3. Open Setup_InputFormat.ipynb notebook"
echo "4. Select the \"$PYTHON_KERNEL_NAME\": Top Menu -> Kernel -> Change Kernel"

# End of the script
